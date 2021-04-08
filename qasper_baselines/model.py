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

from allennlp_models.rc.metrics import SquadEmAndF1


@Model.register("qasper_baseline")
class QasperBaseline(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model_name: str,
        attention_dropout: float = 0.1,
        evidence_feedforward: FeedForward = None,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        transformer_config = AutoConfig.from_pretrained(transformer_model_name)
        transformer_config.attention_dropout = attention_dropout
        self.transformer = AutoModelForSeq2SeqLM.from_config(transformer_config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name,
            add_special_tokens=False
        )
        if evidence_feedforward:
            self.evidence_feedforward = evidence_feedforward
            assert evidence_feedforward.get_output_dim() == 2
        else:
            self.evidence_feedforward = torch.nn.Linear(
                self.transformer.config.hidden_size, 2
            )
        self._answer_metrics = SquadEmAndF1()
        self._evidence_f1 = Average()

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
                    max_length=100
                )
                predicted_answers = [
                    self.tokenizer.decode(generated_token_ids[i].tolist(), skip_special_tokens=True)
                    for i in range(generated_token_ids.size(0))
                ]
                output_dict["predicted_answers"] = predicted_answers
                gold_answers = [instance_metadata["all_answers"] for instance_metadata in metadata]
                for predicted_answer, gold_answer in zip(predicted_answers, gold_answers):
                    self._answer_metrics(predicted_answer, gold_answer)

        if evidence is not None:
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
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            evidence_loss = loss_fn(evidence_logits.view(-1, 2), evidence.view(-1))
            if loss is None:
                loss = evidence_loss
            else:
                loss = loss + evidence_loss
            if not self.training:
                predicted_evidence_indices = evidence_logits.argmax(dim=-1).tolist()
                gold_evidence_indices = [instance_metadata["all_evidence_masks"]
                                         for instance_metadata in metadata]
                for evidence_f1 in self._compute_evidence_f1(predicted_evidence_indices,
                                                             gold_evidence_indices):
                    self._evidence_f1(evidence_f1)

        return {"loss": loss}

    @staticmethod
    def _compute_evidence_f1(
        predicted_evidence_indices: List[List[int]],
        gold_evidence_indices: List[List[List[int]]]
    ) -> List[float]:
        f1s = []
        for instance_predicted, instance_gold in zip(predicted_evidence_indices, gold_evidence_indices):
            instance_f1s = []
            for gold in instance_gold:
                # If the document was truncated to fit in the model, the gold will be longer than the 
                # predicted indices.
                predicted = instance_predicted + [0] * (len(gold) - len(instance_predicted))
                true_positives = sum([i and j for i, j in zip(predicted, gold)])
                precision = true_positives / sum(predicted) if sum(predicted) != 0 else 0.0
                recall = true_positives / sum(gold) if sum(gold) != 0 else 0.0
                if precision + recall == 0:
                    instance_f1s.append(0.0)
                else:
                    instance_f1s.append(2 * precision * recall / (precision + recall))
            f1s.append(max(instance_f1s))
        return f1s

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        _, f1_score = self._answer_metrics.get_metric(reset)
        evidence_f1 = self._evidence_f1.get_metric(reset)
        return {"answer_f1": f1_score, "evidence_f1": evidence_f1}
