import json
import sys
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

random.seed(31045)

data = json.load(open(sys.argv[1]))

def compute_paragraph_f1(predicted_paragraphs, gold_paragraphs):
    f1s = []
    for reference_paragraphs in gold_paragraphs:
        intersection = set(predicted_paragraphs).intersection(reference_paragraphs)
        if not intersection:
            return 0.0
        precision = 1.0  # since there is only one predicted paragraph
        # Since the baselines cannot select floats, we ignore them. The LED model also
        # ingores evidence in floats.
        recall = 1 / len([p for p in reference_paragraphs if "FLOAT SELECTED" not in p])
        f1s.append(2 * precision * recall / (precision + recall))
    return max(f1s)

random_paragraphs = []
first_paragraphs = []
tfidf_paragraphs = []
gold_evidence = []
for _, paper_data in data.items():
    paragraphs = []
    for section_info in paper_data["full_text"]:
        paragraphs.extend(section_info["paragraphs"])

    paper_vectorizer = TfidfVectorizer(decode_error='replace',
                                       strip_accents='unicode',
                                       analyzer='word',
                                       stop_words='english')
    index = paper_vectorizer.fit_transform(paragraphs)
    for qa_info in paper_data["qas"]:
        question = qa_info["question"]
        gold_evidence.append([answer_info["answer"]["evidence"] for answer_info in qa_info["answers"]])
        first_paragraphs.append([paragraphs[0]])
        random_paragraphs.append([random.choice(paragraphs)])
        query_vector = paper_vectorizer.transform([question])
        similarities = cosine_similarity(index, query_vector).flatten()
        most_similar_index = np.argsort(similarities, axis=0)[-1]
        tfidf_paragraphs.append([paragraphs[most_similar_index]])

assert len(gold_evidence) == len(random_paragraphs)
assert len(gold_evidence) == len(first_paragraphs)
assert len(gold_evidence) == len(tfidf_paragraphs)

random_f1s = []
first_f1s = []
tfidf_f1s = []

for first_paragraph, random_paragraph, tfidf_paragraph, evidence in zip(
                first_paragraphs,
                random_paragraphs,
                tfidf_paragraphs,
                gold_evidence):
    random_f1s.append(compute_paragraph_f1(random_paragraph, evidence))
    first_f1s.append(compute_paragraph_f1(first_paragraph, evidence))
    tfidf_f1s.append(compute_paragraph_f1(tfidf_paragraph, evidence))

print("\nRandom paragraph baseline:", np.mean(random_f1s))

print("\nFirst paragraph baseline:", np.mean(first_f1s))

print("\nTFIDF baseline:", np.mean(tfidf_f1s))
