import nltk
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
import spacy

# Load the SPACY English model
nlp = spacy.load("en_core_web_sm")


path_to_reference = 'Dataset/COCOcaptions.txt'
path_to_model = 'model/Decoder/Generated_Captions.txt'

with open(path_to_model) as f:
    model_data = f.readlines()
model_filenames = [caps.split('\t')[0] for caps in model_data]
model_captions = [caps.replace('\n', '').split('\t')[1] for caps in model_data]

with open(path_to_reference, 'r') as f:
    ref_data = f.readlines()
reference_filenames = [caps.split('\t')[0].split('#')[0] for caps in ref_data]
reference_captions = [caps.replace('\n', '').split('\t')[1] for caps in ref_data]

df = pd.DataFrame()
df['image'] = reference_filenames
df['caption'] = reference_captions
df['caption'] = df['caption'].astype(str).str.split()
df = pd.DataFrame(data={'image': list(df.image.unique()), 'caption': list(df.groupby('image')['caption'].apply(list))})[:len(model_captions)]

bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []
index1 = None
index2 = None

for i, row in df.iterrows():
    model = model_captions[i].split()
    reference = row.caption
    try:
        score1 = nltk.translate.bleu_score.sentence_bleu(reference, model, weights=[1.0])
        score2 = nltk.translate.bleu_score.sentence_bleu(reference, model, weights=[0.5, 0.5])
        score3 = nltk.translate.bleu_score.sentence_bleu(reference, model, weights=[1.0/3, 1.0/3, 1-2*(1.0/3)])
        score4 = nltk.translate.bleu_score.sentence_bleu(reference, model)
        bleu1_scores.append(score1)
        bleu2_scores.append(score2)
        bleu3_scores.append(score3)
        bleu4_scores.append(score4)
        if i % 10000 == 0 and i != 0:
            print(float(i) / df.shape[0]) * 100, "%", " done"
    except:
        index1 = df.index[i]
        index2 = i
        print("Invalid Caption Generated for: ", model_filenames[i])

print("\nMean Sentence-Level BLEU-1 score: ", np.mean(bleu1_scores))
print("Mean Sentence-Level BLEU-2 score: ", np.mean(bleu2_scores))
print("Mean Sentence-Level BLEU-3 score: ", np.mean(bleu3_scores))
print("Mean Sentence-Level BLEU-4 score: ", np.mean(bleu4_scores))

if index1 and index2:
    df = df.drop([index1])
    df = df.reset_index(drop=True)
    del model_captions[index2]

references = df.caption
model_captions = [caption.split() for caption in model_captions]

score1 = nltk.translate.bleu_score.corpus_bleu(references, model_captions, weights=[1.0])
print("\n\nCorpus-Level BLEU-1 score: ", score1)
score2 = nltk.translate.bleu_score.corpus_bleu(references, model_captions, weights=[0.5, 0.5])
print("Corpus-Level BLEU-2 score: ", score2)
score3 = nltk.translate.bleu_score.corpus_bleu(references, model_captions, weights=[1.0/3, 1.0/3, 1-2*(1.0/3)])
print("Corpus-Level BLEU-3 score: ", score3)
score4 = nltk.translate.bleu_score.corpus_bleu(references, model_captions)
print("Corpus-Level BLEU-4 score: ", score4)

##################################### add othmane sebban rouge-l #########################################
# Compute Rouge-L scores
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_scores = []

for i, row in df.iterrows():
    model = ' '.join(model_captions[i])  # Convert the list to a string
    reference = ' '.join(str(word) for word in row.caption)  # Convert the list to a string
    try:
        rouge_score = scorer.score(model, reference)['rougeL'].fmeasure
        rouge_scores.append(rouge_score)
    except:
        print("Invalid Caption Generated for: ", model_filenames[i])

print("\nMean Rouge-L score: ", np.mean(rouge_scores))


##################################### add othmane sebban spicy #########################################
def compute_spice(reference, model):
    try:
        reference_tokens = nltk.word_tokenize(reference)
        model_tokens = nltk.word_tokenize(model)
        # Compute similarity using SPICE metric
        common_tokens = set(reference_tokens).intersection(model_tokens)
        precision = len(common_tokens) / len(model_tokens) if len(model_tokens) > 0 else 0
        recall = len(common_tokens) / len(reference_tokens) if len(reference_tokens) > 0 else 0
        # Compute SPICE score
        spice_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        # Debugging: Print relevant values
        print("Precision:", precision)
        print("Recall:", recall)
        print("Common Tokens:", common_tokens)
        return spice_score

    except Exception as e:
        print("Error in compute_spice:", e)
        return 0
# Compute SPICE scores
spice_scores = []
for i, row in df.iterrows():
    model = ' '.join(model_captions[i])
    reference = ' '.join(str(word) for word in row.caption)
    try:
        spice_score = compute_spice(reference, model)
        spice_scores.append(spice_score)
    except:
        print("Invalid Caption Generated for: ", model_filenames[i])
print("\nMean SPICE score: ", np.mean(spice_scores))

output_file = 'result.txt'
with open(output_file, 'w') as f:
    f.write(f"Corpus-Level BLEU-1 score: {score1}\n")
    f.write(f"Corpus-Level BLEU-2 score: {score2}\n")
    f.write(f"Corpus-Level BLEU-3 score: {score3}\n")
    f.write(f"Corpus-Level BLEU-4 score: {score4}\n")
    f.write(f"Mean Rouge-L score: {np.mean(rouge_scores)}\n")
    f.write(f"Mean SPICE score: {np.mean(spice_scores)}\n")

print(f"Corpus-level BLEU and CIDEr scores saved to {output_file}")
