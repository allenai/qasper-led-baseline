import argparse
import sys
import os
import json
from tqdm import tqdm
import torch
from allennlp.models.archival import load_archive
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qasper_baselines import model, dataset_reader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--output', type=str)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    archive = load_archive(args.model)
    qasper_led = archive.model.transformer
    if not args.cpu:
        qasper_led.cuda()
    tokenizer = AutoTokenizer.from_pretrained('allenai/led-base-16384')
    reader = dataset_reader.QasperReader(for_training=False)
    dataset = json.load(open(args.data))
    outfile = open(args.output, "w")
    for article_id, article in tqdm(dataset.items()):
        article["article_id"] = article_id
        for instance in reader._article_to_instances(article):
            question_id = instance.fields['metadata'].metadata['question_id']
            try:
                tokens = instance['question_with_context'].human_readable_repr()
                global_attention_mask = torch.tensor([list(instance['global_attention_mask'].array)])
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids = torch.tensor([token_ids])
                attention_mask = torch.tensor([[True] * len(token_ids)])
                if not args.cpu:
                    global_attention_mask = global_attention_mask.cuda()
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                generation_output = qasper_led.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        global_attention_mask=global_attention_mask,
                        do_sample=True,
                        num_return_sequences=args.samples,
                        output_scores=True,
                        return_dict_in_generate=True
                )
                answers = []
                answer_token_log_probs = []
                normalized_answer_log_probs = []
                output_sequences = generation_output.sequences.tolist()
                output_scores = generation_output.scores
                for answer_id, sequence in enumerate(output_sequences):
                    answers.append(tokenizer.decode(sequence, skip_special_tokens=True))
                    answer_token_log_probs.append([])
                    word_pieces = tokenizer.convert_ids_to_tokens(sequence)
                    # Skipping the first token id because that is the sep token, and the scores start from the
                    # second token.
                    for token_id, token, token_scores in zip(sequence[1:], word_pieces[1:], output_scores):
                        if token == "<pad>":
                            break
                        token_log_prob = torch.log(torch.softmax(token_scores[answer_id], 0)[token_id]).tolist()
                        answer_token_log_probs[-1].append(token_log_prob)
                    normalized_answer_log_probs.append(sum(answer_token_log_probs[-1]) / len(answer_token_log_probs[-1]))

                output_data = {
                        "question_id": question_id,
                        "answers": answers,
                        "answer_token_log_probs": answer_token_log_probs,
                        "normalized_answer_log_probs": normalized_answer_log_probs
                }
                print(json.dumps(output_data), file=outfile)
                outfile.flush()
            except:
                print(f"Skipping {question_id}")

if __name__ == "__main__":
    main()
