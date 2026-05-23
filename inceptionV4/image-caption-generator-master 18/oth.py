
from cider_scorer import CiderScorer

# ... (your existing code)

cider_scores = []

# Assuming references is a list of lists (each sublist corresponds to a reference for a particular image)
references = [[ref.split() for ref in cap_list] for cap_list in references]

cider_scorer = Cider()
for i, row in df.iterrows():
    model = ' '.join(model_captions[i])  # Convert the list to a string
    reference = [' '.join(str(word) for word in cap) for cap in references[i]]  # Convert the lists to strings
    try:
        cider_score_val, _ = cider_scorer.compute_score({i: [reference]}, {i: model})
        cider_scores.append(cider_score_val)
    except Exception as e:
        print("Invalid Caption Generated for: ", model_filenames[i])
        print(e)

print("\nMean CIDEr score: ", np.mean(cider_scores))
