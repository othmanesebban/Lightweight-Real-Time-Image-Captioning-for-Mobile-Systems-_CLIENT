# convfeatures_resnet50.py
"""
Extract ResNet50 (ImageNet) global-average-pooled features.

Output shape:
- (N, 2048) using include_top=False + pooling='avg'

Requirements:
- TensorFlow 1.x OR TF2 in compat v1 mode (we disable eager).
"""

import os
import argparse
from typing import Iterator, List

import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("Please install pillow: pip install pillow") from e

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


BATCH_SIZE = 10
IMAGE_SIZE = 224
OUTPUT_SIZE = 2048


def list_images(img_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png"}
    files = []
    for name in os.listdir(img_dir):
        _, ext = os.path.splitext(name.lower())
        if ext in exts:
            files.append(name)
    return sorted(files)


def load_image_np(path: str) -> np.ndarray:
    """
    Returns:
        np.ndarray float32 of shape (224, 224, 3) in RGB order, range [0..255].
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    return arr


def batch_iterator(img_dir: str, files: List[str], batch_size: int) -> Iterator[np.ndarray]:
    for i in range(0, len(files), batch_size):
        batch_files = files[i : i + batch_size]
        if len(batch_files) < batch_size:
            break  # keep identical behavior to your original code (drops last partial batch)

        batch = np.stack([load_image_np(os.path.join(img_dir, f)) for f in batch_files], axis=0)
        # ResNet50 preprocess_input expects RGB in 0..255 float, and applies proper normalization
        batch = preprocess_input(batch)
        yield batch


def build_resnet_graph() -> tf.Tensor:
    """
    Build ResNet50 graph and return the output tensor of shape (None, 2048).
    """
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), name="InputImage")
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_tensor=input_ph)
    output = model.output
    output = tf.identity(output, name="Output_Features")  # (None, 2048)
    return input_ph, output


def forward_pass(img_dir: str, out_path: str) -> None:
    files = list_images(img_dir)
    print("#Images:", len(files))

    if len(files) < BATCH_SIZE:
        raise ValueError(f"Not enough images for one batch. Need >= {BATCH_SIZE}, got {len(files)}")

    input_ph, output_tensor = build_resnet_graph()

    n_batch = len(files) // BATCH_SIZE
    feats = []

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for i, batch in enumerate(batch_iterator(img_dir, files, BATCH_SIZE), start=1):
            out = sess.run(output_tensor, feed_dict={input_ph: batch})
            out = out.reshape(BATCH_SIZE, OUTPUT_SIZE)
            feats.append(out)

            if i % 5 == 0 or i == n_batch:
                print(f"Progress: {i}/{n_batch} ({(i / float(n_batch)) * 100:.1f}%)")

    feats = np.concatenate(feats, axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, feats)
    print("Saved features:", out_path + ".npy", "shape:", feats.shape)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to images folder (jpg/png).")
    parser.add_argument("--out_path", type=str, default="Dataset/features_resnet50", help="Output .npy path (without extension).")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    print("Extracting ResNet50 features")
    forward_pass(args.data_path, args.out_path)
    print("done")