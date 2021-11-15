from typing import Any, Dict, List
from overrides import overrides

from transformers import AutoModel, AutoTokenizer
from transformers.models.led.modeling_led import shift_tokens_right
import torch

from allennlp.nn import util
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward
from allennlp.training.metrics import Average

from allennlp_models.rc.tools import squad

from qasper_baselines.dataset_reader import AnswerType


@Model.register("qasper_evidence")
class QasperBaseline(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model_name: str = "roberta-base",
        num_top_candidates_for_evaluation: int = 5,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.query_encoder = AutoModel.from_pretrained(transformer_model_name)
        self.target_encoder = AutoModel.from_pretrained(transformer_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name,
            add_special_tokens=False
        )
        self._top_k_precision = Average()
        self._top_k_recall = Average()
        self._top_k_f1 = Average()
        self._num_candidates_for_evaluation = num_top_candidates_for_evaluation
        self._loss_function = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        query: TextFieldTensors,
        target_candidates: TextFieldTensors,
        target_index: torch.Tensor = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, torch.Tensor]:
        query_ids = util.get_token_ids_from_text_field_tensors(query)
        query_attention_mask = util.get_text_field_mask(query)

        query_encoder_output = self.query_encoder(
            input_ids=query_ids,
            attention_mask=query_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        # (batch_size, transformer_hidden_size)
        encoded_query = query_encoder_output["pooler_output"]

        target_candidate_ids = util.get_token_ids_from_text_field_tensors(target_candidates)
        target_candidate_attention_mask = util.get_text_field_mask(target_candidates)
        batch_size, num_targets, num_tokens = target_candidate_ids.size()
        target_candidate_ids = target_candidate_ids.view(-1, num_tokens)
        target_candidate_attention_mask = target_candidate_attention_mask.view(-1, num_tokens)
        target_encoder_output = self.target_encoder(
            input_ids=target_candidate_ids,
            attention_mask=target_candidate_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        encoded_target_candidates = target_encoder_output["pooler_output"]
        # (batch_size, num_targets, transformer_hidden_size)
        encoded_target_candidates = encoded_target_candidates.view(batch_size, num_targets, -1)

        # (batch_size, num_targets)
        target_scores = torch.bmm(encoded_target_candidates, encoded_query.unsqueeze(2)).squeeze(-1)
        loss = None
        if target_index is not None:
            target_index = target_index.squeeze(-1)
            loss = self._loss_function(target_scores, target_index)
        else:
            sorted_indices = torch.argsort(target_scores, descending=True)
            for instance_sorted_indices, instance_metadata in zip(sorted_indices, metadata):
                # TODO (pradeep): Also evaluate on predicting NULL evidence.
                if not instance_metadata["all_evidence"]:
                    continue
                precision, recall, f1 = self.compute_top_k_metrics(instance_sorted_indices,
                                                                   instance_metadata["all_paragraphs"],
                                                                   instance_metadata["all_evidence"])
                self._top_k_precision(precision)
                self._top_k_recall(recall)
                self._top_k_f1(f1)

        output_dict = {"loss": loss}
        return output_dict

    def compute_top_k_metrics(self, indices, paragraphs, all_evidence):
        predicted_evidence = [paragraphs[i] for i in indices[:self._num_candidates_for_evaluation]]
        max_f1 = 0.0
        max_precision = 0.0
        max_recall = 0.0
        for evidence in all_evidence:
            overlap = set(predicted_evidence).intersection(evidence)
            precision = len(overlap) / self._num_candidates_for_evaluation
            recall = len(overlap) / len(evidence)
            f1 = 2 * precision * recall / (precision + recall) if not (precision + recall) == 0 else 0.0
            if f1 > max_f1:
                max_f1 = f1
                max_precision = precision
                max_recall = recall
        return max_precision, max_recall, max_f1

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "top_k_precision": self._top_k_precision.get_metric(reset),
            "top_k_recall": self._top_k_recall.get_metric(reset),
            "top_k_f1": self._top_k_f1.get_metric(reset),
        }
