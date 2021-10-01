# pylint: disable=no-self-use,invalid-name
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary

from qasper_baselines.evidence_retrieval_reader import QasperEvidenceReader


class TestQasperEvidenceReader:
    def test_read_from_file(self):
        reader = QasperEvidenceReader(max_num_negatives=2)
        instances = ensure_list(reader.read("fixtures/data/qasper_sample_small.json"))
        assert len(instances) == 5

        instance = instances[1]
        assert set(instance.fields.keys()) == {
            "query",
            "target_candidates",
            "target_index",
            "metadata",
        }

        query_text = [t.text for t in instance.fields["query"].tokens]
        assert len(query_text) == 18
        assert query_text == [
            '<s>',
            'This',
            'Ġis',
            'Ġthe',
            'Ġabstract',
            'Ġof',
            'Ġthe',
            'Ġpaper',
            '</s>',
            '</s>',
            'What',
            'Ġis',
            'Ġthe',
            'Ġseed',
            'Ġlex',
            'icon',
            '?',
            '</s>'
        ]

        target_candidate_fields = instance["target_candidates"].field_list
        assert len(target_candidate_fields) == 3
        assert [t.text for t in target_candidate_fields[0].tokens] == [
            '<s>',
            'Sh',
            'orter',
            'Ġparagraph',
            '</s>'
        ]

        assert instance["target_index"].sequence_index == 1

        assert [t.text for t in target_candidate_fields[1].tokens] == [
            '<s>',
            'Method',
            'Ġparagraph',
            'Ġusing',
            'Ġseed',
            'Ġlex',
            'icon',
            '</s>'
        ]

        assert [t.text for t in target_candidate_fields[2].tokens] == [
            '<s>',
            'A',
            'Ġshort',
            'Ġparagraph',
            '</s>'
        ]

        assert instance["metadata"].keys() == {
            "question",
            "abstract",
            "query_tokens",
            "target_candidates",
            "all_evidence",
            "all_paragraphs",
            "article_id",
            "question_id",
        }

    def test_read_from_file_for_test(self):
        reader = QasperEvidenceReader(max_num_negatives=2, for_training=False)
        instances = ensure_list(reader.read("fixtures/data/qasper_sample_small.json"))
        assert len(instances) == 4  # Only one instance per question

        instance = instances[0]
        assert set(instance.fields.keys()) == {
            "query",
            "target_candidates",
            "metadata",
        }

        query_text = [t.text for t in instance.fields["query"].tokens]
        assert len(query_text) == 18

        target_candidate_fields = instance["target_candidates"].field_list
        assert len(target_candidate_fields) == 5
