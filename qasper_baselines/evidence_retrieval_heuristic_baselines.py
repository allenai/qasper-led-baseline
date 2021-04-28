import json
import sys
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from allennlp.training.metrics import Average

random.seed(31045)

data = json.load(open(sys.argv[1]))


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
        evidence = []
        for answer_info in qa_info["answers"]:
            evidence.extend(answer_info["answer"]["evidence"])
        gold_evidence.append(evidence)
        first_paragraphs.append([paragraphs[0]])
        random_paragraphs.append([random.choice(paragraphs)])
        query_vector = paper_vectorizer.transform([question])
        similarities = cosine_similarity(index, query_vector).flatten()
        most_similar_index = np.argsort(similarities, axis=0)[-1]
        tfidf_paragraphs.append([paragraphs[most_similar_index]])

random_baseline_metric = Average()
first_baseline_metric = Average()
tfidf_baseline_metric = Average()

assert len(gold_evidence) == len(random_paragraphs)
assert len(gold_evidence) == len(first_paragraphs)
assert len(gold_evidence) == len(tfidf_paragraphs)

for first_paragraph, random_paragraph, tfidf_paragraph, evidence in zip(
                first_paragraphs,
                random_paragraphs,
                tfidf_paragraphs,
                gold_evidence):
    random_baseline_metric(1.0 if random_paragraph[0] in evidence else 0.0)
    first_baseline_metric(1.0 if first_paragraph[0] in evidence else 0.0)
    tfidf_baseline_metric(1.0 if tfidf_paragraph[0] in evidence else 0.0)

print("Random paragraph baseline:")
print(random_baseline_metric.get_metric(reset=True))

print("First paragraph baseline:")
print(first_baseline_metric.get_metric(reset=True))

print("TFIDF baseline:")
print(tfidf_baseline_metric.get_metric(reset=True))
