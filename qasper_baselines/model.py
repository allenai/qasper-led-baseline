import os
import logging
from typing import Any, Dict, List
from overrides import overrides

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.led.modeling_led import shift_tokens_right
import torch

from allennlp.nn import util
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.training.metrics import Average

from allennlp_models.rc.tools import squad

from qasper_baselines.dataset_reader import AnswerType
# from sci_long_t5.model import LongT5Config, LongT5TokenizerFast, LongT5ForConditionalGeneration

logger = logging.getLogger(__name__)

@Model.register("qasper_baseline")
class QasperBaseline(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model_name: str,
        attention_dropout: float = 0.1,
        attention_window_size: int = 1024,
        gradient_checkpointing: bool = False,
        evidence_feedforward: FeedForward = None,
        use_only_evidence_loss: bool = False,
        use_evidence_scaffold: bool = True,
        use_margin_loss_for_evidence: bool = False,
        per_reference_level_metrics: bool = False,
        resume_model_dir: str = None,
        resume_model_file: str = None,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        config = AutoConfig.from_pretrained(transformer_model_name)
        config.attention_dropout = attention_dropout
        config.attention_window = [attention_window_size] * len(config.attention_window)
        config.gradient_checkpointing = gradient_checkpointing
        if resume_model_dir is not None:
            led_model = torch.load(os.path.join(resume_model_dir, resume_model_file))
            renamed_state_dict = {}
            for k, v in led_model["state_dict"].items():
                new_key = k.replace("model.led.", "")
                renamed_state_dict[new_key] = v
            self.transformer = AutoModelForSeq2SeqLM.from_pretrained(None, config=config, state_dict=renamed_state_dict)
        else:
            self.transformer = AutoModelForSeq2SeqLM.from_pretrained(transformer_model_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name,
            add_special_tokens=False
        )

        if evidence_feedforward:
            self.evidence_feedforward = evidence_feedforward
            assert evidence_feedforward.get_output_dim() == 2
        else:
            if use_margin_loss_for_evidence:
                self.evidence_feedforward = torch.nn.Linear(
                    self.transformer.config.hidden_size, 1
                )
            else:
                self.evidence_feedforward = torch.nn.Linear(
                    self.transformer.config.hidden_size, 2
                )
        self._use_only_evidence_loss = use_only_evidence_loss
        self._use_evidence_scaffold = use_evidence_scaffold
        self._use_margin_loss_for_evidence = use_margin_loss_for_evidence
        self._per_reference_level_metrics = per_reference_level_metrics
        self._answer_f1 = Average()
        self._answer_f1_by_type = {answer_type: Average() for answer_type in AnswerType}
        self._evidence_f1 = Average()
        self._evidence_loss = Average()
        self._train_evidence_logit_thresh = Average()

    def forward(
        self,
        question_with_context: TextFieldTensors,
        paragraph_indices: torch.Tensor,
        global_attention_mask: torch.Tensor = None,
        evidence: torch.Tensor = None,
        answer: TextFieldTensors = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, torch.Tensor]:
        input_ids = util.get_token_ids_from_text_field_tensors(question_with_context)
        attention_mask = util.get_text_field_mask(question_with_context)

        if answer is not None:
            answer_ids = util.get_token_ids_from_text_field_tensors(answer)
        else:
            answer_ids = None

        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            labels=answer_ids,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        encoded_tokens = output["encoder_last_hidden_state"]

        output_dict = {}
        output_dict["answer_logits"] = output["logits"]
        loss = None
        if answer is not None:
            loss = output['loss']
            if not self.training:
                # Computing evaluation metrics
                # max_length: 100 covers 97% of the data. 116 for 98%, 169 for 99%, 390 for 99.9%, 
                # 606 for 100%
                generated_token_ids = self.transformer.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    max_length=100
                )
                predicted_answers = [
                    self.tokenizer.decode(generated_token_ids[i].tolist(), skip_special_tokens=True)
                    for i in range(generated_token_ids.size(0))
                ]
                output_dict["predicted_answers"] = predicted_answers
                gold_answers = [instance_metadata["all_answers"] for instance_metadata in metadata]
                for predicted_answer, gold_answer in zip(predicted_answers, gold_answers):
                    f1s_with_types = []
                    for gold_answer_info in gold_answer:
                        f1 = squad.compute_f1(predicted_answer, gold_answer_info['text'])
                        f1s_with_types.append((f1, gold_answer_info['type']))

                        if self._per_reference_level_metrics:
                            self._answer_f1(f1)
                            self._answer_f1_by_type[gold_answer_info['type']](f1)

                    if not self._per_reference_level_metrics:
                        max_f1, max_f1_answer_type = sorted(f1s_with_types, key=lambda x: x[0])[-1]
                        self._answer_f1(max_f1)
                        self._answer_f1_by_type[max_f1_answer_type](max_f1)

        if self._use_evidence_scaffold and evidence is not None:
            paragraph_indices = paragraph_indices.squeeze(-1)
            encoded_paragraph_tokens = util.batched_index_select(encoded_tokens.contiguous(), paragraph_indices)
            evidence_logits = self.evidence_feedforward(encoded_paragraph_tokens)
            evidence_mask = paragraph_indices != -1

            # Use a loss function that gives higher weight to the positive classes
            weights = torch.tensor(
                [
                    evidence.sum() + 1,
                    evidence_mask.sum() - evidence.sum() + 1,
                ],
                device=evidence_logits.device,
                dtype=evidence_logits.dtype,
            )
            if self._use_margin_loss_for_evidence:
                positive_example_indices = (evidence == 1).nonzero()[:, 1].unsqueeze(0)
                positive_example_scores = util.batched_index_select(evidence_logits, positive_example_indices).squeeze(0)
                min_positive_score = torch.min(positive_example_scores).unsqueeze(0)

                negative_example_indices = (evidence == 0).nonzero()[:, 1].unsqueeze(0)
                negative_example_scores = util.batched_index_select(evidence_logits, negative_example_indices).squeeze(0)
                max_negative_score = torch.max(negative_example_scores).unsqueeze(0)

                if self.training:
                    min_positive_index = positive_example_indices[:, torch.argmin(positive_example_scores)]
                    evidence_probs = torch.softmax(evidence_logits, dim=1)
                    min_positive_prob = evidence_probs[:, min_positive_index, :]
                    max_negative_index = negative_example_indices[:, torch.argmin(negative_example_scores)]
                    max_negative_prob = evidence_probs[:, max_negative_index, :]
                    if max_negative_prob < min_positive_prob:
                        self._train_evidence_logit_thresh(max_negative_prob+(min_positive_prob-max_negative_prob)/2)
                    else:
                        self._train_evidence_logit_thresh(min_positive_prob+(max_negative_prob-min_positive_prob)/2)

                loss_fn = torch.nn.MarginRankingLoss(margin=0.5)
                evidence_loss = loss_fn(min_positive_score, max_negative_score,
                                        torch.ones(min_positive_score.size(), device=max_negative_score.device))
                self._evidence_loss(float(evidence_loss.detach().cpu()))
            else:
                loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
                evidence_loss = loss_fn(evidence_logits.view(-1, 2), evidence.view(-1))
                self._evidence_loss(float(evidence_loss.detach().cpu()))

            if loss is None:
                loss = evidence_loss
            else:
                if self._use_only_evidence_loss:
                    loss = evidence_loss
                else:
                    loss = loss + evidence_loss
            if not self.training:
                if self._use_margin_loss_for_evidence:
                    # predicted_evidence_scores = evidence_logits.tolist()[1:]
                    predicted_evidence_scores = evidence_probs[:, 1:, ].reshape((1, evidence.size(1)-1))
                    threshold = self._train_evidence_logit_thresh.get_metric(reset=False)
                    predicted_evidence_indices = (predicted_evidence_scores > threshold).int().tolist()
                else:
                    predicted_evidence_indices = evidence_logits.argmax(dim=-1).tolist()
                gold_evidence_indices = [instance_metadata["all_evidence_masks"]
                                         for instance_metadata in metadata]
                for evidence_f1 in self._compute_evidence_f1(predicted_evidence_indices,
                                                             gold_evidence_indices):
                    if self._per_reference_level_metrics:
                        for ref_evidence_f1 in evidence_f1:
                            self._evidence_f1(ref_evidence_f1)
                    else:
                        self._evidence_f1(max(evidence_f1))
        output_dict["loss"] = loss
        return output_dict

    @staticmethod
    def _compute_evidence_f1(
        predicted_evidence_indices: List[List[int]],
        gold_evidence_indices: List[List[List[int]]]
    ) -> List[List[float]]:
        f1s = []
        for instance_predicted, instance_gold in zip(predicted_evidence_indices, gold_evidence_indices):
            instance_f1s = []
            for gold in instance_gold:
                # If the document was truncated to fit in the model, the gold will be longer than the 
                # predicted indices.
                predicted = instance_predicted + [0] * (len(gold) - len(instance_predicted))
                true_positives = sum([i and j for i, j in zip(predicted, gold)])
                if sum(predicted) == 0:
                    precision = 1.0 if sum(gold) == 0 else 0.0
                else:
                    precision = true_positives / sum(predicted)
                recall = true_positives / sum(gold) if sum(gold) != 0 else 1.0
                if precision + recall == 0:
                    instance_f1s.append(0.0)
                else:
                    instance_f1s.append(2 * precision * recall / (precision + recall))
            f1s.append(instance_f1s)
            # if self._per_reference_level_metrics:
            #     f1s.extend(instance_f1s)
            # else:
            #     f1s.append(max(instance_f1s))
        return f1s

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_score = self._answer_f1.get_metric(reset)
        extractive_f1_score = self._answer_f1_by_type[AnswerType.EXTRACTIVE].get_metric(reset)
        abstractive_f1_score = self._answer_f1_by_type[AnswerType.ABSTRACTIVE].get_metric(reset)
        boolean_f1_score = self._answer_f1_by_type[AnswerType.BOOLEAN].get_metric(reset)
        none_f1_score = self._answer_f1_by_type[AnswerType.NONE].get_metric(reset)
        evidence_f1 = self._evidence_f1.get_metric(reset)
        evidence_loss = self._evidence_loss.get_metric(reset)
        threshold = self._train_evidence_logit_thresh.get_metric(reset=True)
        return {
            "answer_f1": f1_score,
            "extr_f1": extractive_f1_score,
            "abstr_f1": abstractive_f1_score,
            "bool_f1": boolean_f1_score,
            "none_f1": none_f1_score,
            "evidence_f1": evidence_f1,
            "evidence_loss": evidence_loss
        }
