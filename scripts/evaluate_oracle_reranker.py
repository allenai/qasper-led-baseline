import json
import argparse
from allennlp_models.rc.metrics import SquadEmAndF1

metric = SquadEmAndF1()
f1_hash = {}
def get_f1(pred, ans):
    pred = pred.strip()
    ans = ans.strip()
    if (pred, ans) in f1_hash:
        return f1_hash[(pred, ans)]
    if (ans, pred) in f1_hash:
        return f1_hash[(ans, pred)]
    metric(pred, [ans])
    _, f1 = metric.get_metric(True)
    f1_hash[(pred, ans)] = f1
    return f1


def get_references(answers_info):
    references = []
    for answer_info in answers_info:
        answer = answer_info["answer"]
        if answer["unanswerable"]:
            references.append("Unanswerable")
        elif answer["extractive_spans"]:
            references.append(", ".join(answer["extractive_spans"]))
        elif answer["free_form_answer"]:
            references.append(answer["free_form_answer"])
        else:
            references.append("Yes" if answer["yes_no"] else "No")
    return references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--samples", type=str)
    parser.add_argument("--log", type=str)
    args = parser.parse_args()

    data = json.load(open(args.data))
    answers = {}
    questions = {}
    for paper_info in data.values():
        for qa_info in paper_info["qas"]:
            answers[qa_info["question_id"]] = get_references(qa_info["answers"])
            questions[qa_info["question_id"]] = qa_info["question"]

    samples_data = [json.loads(line) for line in open(args.samples)]
    print(f"Read {len(samples_data)} predictions")
    oracle_f1_ranks = []
    oracle_f1s = []
    model_f1s = []
    log_file = open(args.log, "w") if args.log else None
    for prediction_info in samples_data:
        references = answers[prediction_info["question_id"]]
        predictions = prediction_info["answers"]
        scores = prediction_info["normalized_answer_log_probs"]
        sorted_predictions = [y[1] for y in sorted(zip(scores, predictions), key=lambda x: x[0], reverse=True)]
        f1s = [max([get_f1(pred, reference) for reference in references]) for pred in sorted_predictions]
        oracle_f1 = max(f1s)
        model_f1s.append(f1s[0])
        oracle_f1s.append(oracle_f1)
        for i, f1 in enumerate(f1s):
            if f1 == oracle_f1:
                oracle_f1_ranks.append(i+1)
                break
        if log_file:
            print(
                    json.dumps(
                        {
                            "question_id": prediction_info["question_id"],
                            "question": questions[prediction_info["question_id"]],
                            "references": references,
                            "best_model_answer": sorted_predictions[0],
                            "oracle_answer": sorted_predictions[i],
                            "oracle_answer_rank": i+1
                        }
                    ),
                    file=log_file
            )

    average = lambda l: sum(l) / len(l)
    print(f"Average oracle F1 rank: {average(oracle_f1_ranks)}")
    print(f"Average model F1: {average(model_f1s)}")
    print(f"Average oracle F1: {average(oracle_f1s)}")


if __name__ == "__main__":
    main()
