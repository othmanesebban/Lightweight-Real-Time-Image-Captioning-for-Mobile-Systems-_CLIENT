# utils/data_util.py

import numpy as np
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
import pickle
import json
import os

max_len = 16
word_threshold = 2
counter = None


def prepare_coco_captions(filename="C:/Users/LENOVO/Dataset/captions_val2017.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    images = data["images"]
    captions = data["annotations"]
    prefix = "COCO_train2017_"

    for cap in captions:
        image_id = str(cap["image_id"])
        len_id = len(image_id)
        zeros = "0" * (12 - len_id)
        image_id = prefix + zeros + image_id
        cap["image_id"] = image_id
        cap["caption"] = (
            cap["caption"]
            .replace("\n", "")
            .replace(",", " ,")
            .replace(".", "")
            .replace('"', '" ')
            .replace("'s", " 's")
            .replace("'t", " 't")
            + " ."
        )

    captions = sorted(captions, key=lambda k: k["image_id"])
    cap_path = "Dataset/COCOcaptions.txt"

    with open(cap_path, "w") as f:
        for i, cap in enumerate(captions):
            f.write(cap["image_id"] + "#" + str(i % 5) + "\t" + cap["caption"] + "\n")

    return cap_path


def preprocess_coco_captions(filenames, captions):
    df = pd.DataFrame()
    df["FileNames"] = filenames
    df["caption"] = captions
    df["caption"] = (
        df["caption"]
        .apply(word_tokenize)
        .apply(lambda x: x[:16])
        .apply(" ".join)
        .str.lower()
    )

    anomalies = df.FileNames.value_counts()[(df.FileNames.value_counts() > 5)].index.tolist()

    for name in anomalies:
        indexes = df[df.FileNames == name].index[5:]
        df = df.drop(indexes)
        df = df.reset_index(drop=True)

    with open("Dataset/COCOcaptions.txt", "w") as f:
        for i, row in df.iterrows():
            f.write(row["FileNames"] + "#" + str(i % 5) + "\t" + row["caption"] + "\n")

    return df


def preprocess_flickr_captions(filenames, captions):
    global max_len
    print("Preprocessing Captions")

    df = pd.DataFrame()
    df["FileNames"] = filenames
    df["caption"] = captions
    df["caption"] = (
        df["caption"]
        .apply(word_tokenize)
        .apply(lambda x: x[:max_len])
        .apply(" ".join)
        .str.lower()
    )
    return df


def generate_vocab(df):
    global max_len, word_threshold, counter
    print("Generating Vocabulary")

    vocab = dict([w for w in counter.items() if w[1] >= word_threshold])
    vocab["<UNK>"] = len(counter) - len(vocab)
    vocab["<PAD>"] = df.caption.str.count("<PAD>").sum()
    vocab["<S>"] = df.caption.str.count("<S>").sum()
    vocab["</S>"] = df.caption.str.count("</S>").sum()

    wtoidx = {
        "<PAD>": 0,
        "<S>": 1,
        "</S>": 2,
        "<UNK>": 3,
    }

    print("Generating Word to Index and Index to Word")
    i = 4
    for word in vocab.keys():
        if word not in ["<S>", "</S>", "<PAD>", "<UNK>"]:
            wtoidx[word] = i
            i += 1

    print("Size of Vocabulary", len(vocab))
    return vocab, wtoidx


def pad_captions(df, debug=True, debug_samples=5):
    global max_len

    original_max_len = max_len
    padded_max_len = original_max_len + 2

    print("Padding Caption <PAD> to Max Length", original_max_len, "+ 2 for <S> and </S>")

    df_padded = df.copy()
    df_padded["caption"] = "<S> " + df_padded["caption"] + " </S>"

    shown = 0
    for i, row in df_padded.iterrows():
        before_caption = row["caption"]
        before_tokens = before_caption.split()
        before_len = len(before_tokens)

        if before_len < padded_max_len:
            pad_len = padded_max_len - before_len
            pad_buf = " <PAD>" * pad_len
            after_caption = before_caption + pad_buf
            df_padded.at[i, "caption"] = after_caption
        else:
            pad_len = 0
            after_caption = before_caption

        if debug and shown < debug_samples:
            after_tokens = after_caption.split()
            print("\n" + "=" * 100)
            print(f"[DEBUG] Caption #{shown + 1}")
            print("Avant padding :", before_caption)
            print("Tokens avant  :", before_tokens)
            print("Longueur avant:", before_len)
            print("PAD ajoutés   :", pad_len)
            print("Après padding :", after_caption)
            print("Tokens après  :", after_tokens)
            print("Longueur après:", len(after_tokens))
            print("=" * 100)
            shown += 1

    max_len = padded_max_len
    return df_padded


def load_features(feature_path):
    features = np.load(feature_path)
    features = np.repeat(features, 5, axis=0)
    print("Features Loaded", feature_path)
    return features


def split_dataset(df, features, ratio=0.8):
    split_idx = int(df.shape[0] * ratio)
    print("Data Statistics:")
    print("# Records Total Data: ", df.shape[0])
    print("# Records Training Data: ", split_idx)
    print("# Records Validation Data: ", df.shape[0] - split_idx)
    print("Ratio of Training: Validation = ", ratio * 100, ":", 100 - (ratio * 100))

    val_features = features[split_idx:]
    val_captions = np.array(df.caption)[split_idx:]
    np.save("Dataset/Validation_Data", list(zip(val_features, val_captions)))

    return df[:split_idx], features[:split_idx]


def get_data(required_files):
    ret = []
    for fil in required_files:
        ret.append(np.load("Dataset/" + fil + ".npy", allow_pickle=True))
    return ret


def generate_captions(
    wt=2,
    ml=16,
    cap_path="Dataset/results_20130124.token",
    feat_path="Dataset/features.npy",
    data_is_coco=False,
):
    required_files = ["vocab", "wordmap", "Training_Data"]
    generate = False

    for fil in required_files:
        if not os.path.isfile("Dataset/" + fil + ".npy"):
            generate = True
            print("Required Files not present. Regenerating Data.")
            break

    if not generate:
        print("Dataset Present; Skipping Generation.")
        return get_data(required_files)

    global max_len, word_threshold, counter
    max_len = ml
    word_threshold = wt

    print("Loading Caption Data", cap_path)

    if data_is_coco:
        cap_path = prepare_coco_captions(cap_path)
        with open(cap_path, "r", encoding="utf8") as f:
            data = f.readlines()
        filenames = [caps.split("\t")[0].split("#")[0] for caps in data]
        captions = [caps.split("\t")[1] for caps in data]
        df = preprocess_coco_captions(filenames, captions)
    else:
        with open(cap_path, "r", encoding="utf8") as f:
            data = f.readlines()
        filenames = [caps.split("\t")[0].split("#")[0] for caps in data]
        captions = [caps.replace("\n", "").split("\t")[1] for caps in data]
        df = preprocess_flickr_captions(filenames, captions)

    features = load_features(feat_path)
    print("Features shapes:")
    print(features.shape, df.shape)

    idx = np.random.permutation(features.shape[0])
    df = df.iloc[idx].reset_index(drop=True)
    features = features[idx]

    counter = Counter()
    for _, row in df.iterrows():
        counter.update(row["caption"].lower().split())

    df = pad_captions(df, debug=True, debug_samples=5)

    vocab, wtoidx = generate_vocab(df)
    captions = np.array(df.caption)

    np.save("Dataset/Training_Data", list(zip(features, captions)))
    np.save("Dataset/wordmap", wtoidx)
    np.save("Dataset/vocab", vocab)

    print("Preprocessing Complete")
    return get_data(required_files)