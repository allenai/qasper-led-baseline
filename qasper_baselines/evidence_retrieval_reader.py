import json
import logging
import random
from enum import Enum
from collections import defaultdict
from typing import Any, Dict, List, Optional, Iterable, Tuple

from overrides import overrides

import spacy
import torch

from allennlp.common.util import JsonDict
from allennlp.data.fields import (
    MetadataField,
    TextField,
    IndexField,
    ListField,
    TensorField,
)
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer


random.seed(20198)

logger = logging.getLogger(__name__)


@DatasetReader.register("qasper_evidence")
class QasperEvidenceReader(DatasetReader):
    def __init__(
        self,
        transformer_model_name: str = "roberta-base",
        max_query_length: int = 512,
        max_target_length: int = 512,
        max_num_negatives: int = 10,
        for_training: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self._transformer_model_name = transformer_model_name
        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False
        )

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(transformer_model_name)
        }
        self.max_query_length = max_query_length
        self.max_target_length = max_target_length
        self.max_num_negatives = max_num_negatives
        self._for_training = for_training
        self._stats = defaultdict(int)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading json file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        for article_id, article in self.shard_iterable(dataset.items()):
            if not article["full_text"]:
                continue
            article["article_id"] = article_id
            yield from self._article_to_instances(article)
        self._log_stats()

    def _log_stats(self) -> None:
        logger.info("Stats:")
        for key, value in self._stats.items():
            logger.info("%s: %d", key, value)

    def _article_to_instances(self, article: Dict[str, Any]) -> Iterable[Instance]:
        paragraphs = self._get_paragraphs_from_article(article)

        self._stats["number of documents"] += 1
        for question_answer in article["qas"]:
            self._stats["number of questions"] += 1
            self._stats["number of references"] += len(question_answer["answers"])
            if len(question_answer["answers"]) > 1:
                self._stats["questions with multiple references"] += 1

            all_evidence = []
            question = question_answer["question"]
            abstract = article["abstract"]
            tokenized_query = self._tokenize_query(question, abstract)
            for answer_annotation in question_answer["answers"]:
                evidence = self._extract_evidence(
                    answer_annotation["answer"]
                )
                if not evidence:
                    # TODO (pradeep): Introduce a NULL target candidate and make that the positive when there is
                    # no evidence. 
                    self._stats["references with no evidence"] += 1
                    continue
                all_evidence.append(evidence)

            additional_metadata = {
                "question_id": question_answer["question_id"],
                "article_id": article.get("article_id"),
                "all_evidence": all_evidence,
                "all_paragraphs": paragraphs,
            }
            if self._for_training:
                all_non_evidence = self._sample_negatives(all_evidence, paragraphs)
                for evidence, non_evidence in zip(all_evidence, all_non_evidence):
                    for positive, negatives in zip(evidence, non_evidence):
                        positive_index = random.randint(0, len(negatives) - 1)
                        target_candidates = negatives[:positive_index] + [positive] + negatives[positive_index:]
                        yield self.text_to_instance(
                            question,
                            abstract,
                            target_candidates,
                            positive_index,
                            tokenized_query,
                            additional_metadata
                        )
            else:
                target_candidates = paragraphs
                yield self.text_to_instance(
                    question=question,
                    abstract=abstract,
                    target_candidates=paragraphs,
                    tokenized_query=tokenized_query,
                    additional_metadata=additional_metadata,
                )

    @overrides
    def text_to_instance(
        self,  # type: ignore  # pylint: disable=arguments-differ
        question: str,
        abstract: str,
        target_candidates: List[str],
        positive_index: int = None,
        tokenized_query: List[Token] = None,
        additional_metadata: Dict[str, Any] = None,
    ) -> Instance:
        fields = {}

        if tokenized_query is None:
            tokenized_query = self._tokenize_query(question, abstract)
        fields["query"] = TextField(tokenized_query)


        target_candidate_fields_list = []
        for target_candidate in target_candidates:
            tokenized_target = (
                self._tokenizer.single_sequence_start_tokens
                + self._tokenizer.tokenize(target_candidate)
                + self._tokenizer.single_sequence_end_tokens
            )
            target_candidate_fields_list.append(TextField(tokenized_target))
        target_candidates_field = ListField(target_candidate_fields_list)
        fields["target_candidates"] = target_candidates_field
        if positive_index is not None:
            target_index_field = IndexField(positive_index, target_candidates_field)
            fields["target_index"] = target_index_field

        # make the metadata
        metadata = {
            "question": question,
            "abstract": abstract,
            "query_tokens": tokenized_query,
            "target_candidates": target_candidates,
        }
        if additional_metadata is not None:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    def _tokenize_query(
        self,
        question: str,
        abstract: str,
    ) -> List[Token]:
        tokenized_question = self._tokenizer.tokenize(question)
        tokenized_abstract = self._tokenizer.tokenize(abstract)
        allowed_abstract_length = (
            self.max_query_length
            - len(tokenized_question)
            - len(self._tokenizer.sequence_pair_start_tokens)
            - len(self._tokenizer.sequence_pair_mid_tokens)
            - len(self._tokenizer.sequence_pair_end_tokens)
        )
        if len(tokenized_abstract) > allowed_abstract_length:
            self._stats["number of truncated abstracts"] += 1
            tokenized_abstract = tokenized_abstract[:allowed_abstract_length]

        tokenized_query = (
            self._tokenizer.sequence_pair_start_tokens
            + tokenized_abstract
            + self._tokenizer.sequence_pair_mid_tokens
            + tokenized_question
            + self._tokenizer.sequence_pair_end_tokens
        )
        return tokenized_query

    def _extract_evidence(
        self, answer: List[JsonDict]
    ) -> List[str]:
        evidence_spans = [x.replace("\n", " ").strip() for x in answer["evidence"]]
        evidence_spans = [x for x in evidence_spans if x != ""]
        if not evidence_spans:
            self._stats["references with no evidence"] += 1
        # TODO (pradeep): Deal with figures and tables.
        if any(["FLOAT SELECTED" in span for span in evidence_spans]):
            # Ignoring question if any of the selected evidence is a table or a figure.
            self._stats["references with table or figure as evidence"] += 1
        if len(evidence_spans) > 1:
            self._stats["references with multiple evidence spans"] += 1

        return evidence_spans

    def _get_paragraphs_from_article(self, article: JsonDict) -> List[str]:
        full_text = article["full_text"]
        paragraphs = []
        for section_info in full_text:
            for paragraph in section_info["paragraphs"]:
                paragraph_text = paragraph.replace("\n", " ").strip()
                if paragraph_text:
                    paragraphs.append(paragraph_text)
        return paragraphs

    def _sample_negatives(
        self,
        all_evidence: List[List[str]],
        paragraphs: List[str]
    ) -> List[List[List[str]]]:
        """
        Returns a sets of negative candidates, one per evidence, such that each set does not contain any of the
        positive evidence and has at most `max_num_negatives` paragraphs.
        """
        positive_pool = set()
        for evidence in all_evidence:
            for snippet in evidence:
                positive_pool.add(snippet)
        negative_pool = [paragraph for paragraph in paragraphs if paragraph not in positive_pool]
        max_num_negatives = min(len(negative_pool), self.max_num_negatives)
        all_negatives = [[random.sample(negative_pool, max_num_negatives) for _ in evidence]
                         for evidence in all_evidence]
        return all_negatives

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["query"].token_indexers = self._token_indexers
        for field in instance.fields["target_candidates"].field_list:
            field.token_indexers = self._token_indexers
