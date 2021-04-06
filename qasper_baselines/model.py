from typing import Any, Dict

from transformers import AutoModelForSeq2SeqLM
from transformers.models.led.modeling_led import shift_tokens_right
import torch

from allennlp.nn import util
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import FeedForward


@Model.register("qasper_baseline")
class QasperBaseline(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model_name: str,
        evidence_feedforward: FeedForward = None,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(transformer_model_name)
        if evidence_feedforward:
            self.evidence_feedforward = evidence_feedforward
            assert evidence_feedforward.get_output_dim() == 2
        else:
            self.evidence_feedforward = torch.nn.Linear(
                self.transformer.config.hidden_size, 2
            )

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

        loss = None
        if answer is not None:
            loss = output["loss"]

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

        return {"loss": loss}
