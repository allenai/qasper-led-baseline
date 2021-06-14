# pylint: disable=no-self-use,invalid-name
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary

from qasper_baselines.dataset_reader import QasperReader


class TestQasperReader:
    def test_read_from_file(self):
        reader = QasperReader()
        instances = ensure_list(reader.read("fixtures/data/qasper_sample_small.json"))
        assert len(instances) == 4

        instance = instances[1]
        assert set(instance.fields.keys()) == {
            "question_with_context",
            "paragraph_indices",
            "global_attention_mask",
            "evidence",
            "answer",
            "metadata",
        }

        token_text = [t.text for t in instance.fields["question_with_context"].tokens]
        assert len(token_text) == 47
        assert token_text[:15] == [
            '<s>',
            'Are',
            'Ġthere',
            'Ġthree',
            '?',
            '</s>',
            'Introduction',
            '</s>',
            'A',
            'Ġshort',
            'Ġparagraph',
            '</s>',
            'Another',
            'Ġintro',
            'Ġparagraph'
        ]

        assert token_text[-15:] == [
            'ĠPol',
            'arity',
            'ĠFunction',
            '</s>',
            'Method',
            'Ġparagraph',
            'Ġusing',
            'Ġseed',
            'Ġlex',
            'icon',
            '</s>',
            'Conclusion',
            '</s>',
            'Conclusion',
            'Ġparagraph',
        ]


        assert len(instance["paragraph_indices"]) == 10
        # This is the first token after the separator for each paragraph.
        assert [x.sequence_index for x in instance["paragraph_indices"]] == [
            5,
            7,
            11,
            15,
            18,
            22,
            26,
            35,
            42,
            44
        ]

        expected_mask = [False] * 47
        # question tokens and paragraph start indices
        for x in [0, 1, 2, 3, 4, 5, 7, 11, 15, 18, 22, 26, 35, 42, 44]:
            expected_mask[x] = True

        assert instance["global_attention_mask"].tensor.tolist() == expected_mask

        assert len(instance["evidence"]) == len(instance["paragraph_indices"])
        assert instance["evidence"].tensor.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

        answer_text = [t.text for t in instance.fields["answer"].tokens]
        assert answer_text == ["<s>", "Yes", "</s>"]

        assert instance["metadata"].keys() == {
            "question",
            "question_tokens",
            "paragraphs",
            "context_tokens",
            "all_evidence",
            "all_evidence_masks",
            "all_answers",
            "article_id",
            "question_id",
        }

        # Checking the answers for the other instances, which look at different fields of the json.
        answer_text = [t.text for t in instances[0].fields["answer"].tokens]
        assert answer_text == ["<s>", "a", "Ġvocabulary", "</s>"]

        answer_text = [t.text for t in instances[2].fields["answer"].tokens]
        assert answer_text == ["<s>", "Un", "answer", "able", "</s>"]

        answer_text = [t.text for t in instances[3].fields["answer"].tokens]
        assert answer_text == ["<s>", "Conclusion", "Ġparagraph", "</s>"]

    def test_read_from_file_question_only(self):
        reader = QasperReader(context="question_only")
        instances = ensure_list(reader.read("fixtures/data/qasper_sample_small.json"))
        assert len(instances) == 4

        instance = instances[1]
        assert set(instance.fields.keys()) == {
            "question_with_context",
            "paragraph_indices",
            "global_attention_mask",
            "evidence",
            "answer",
            "metadata",
        }

        token_text = [t.text for t in instance.fields["question_with_context"].tokens]
        assert len(token_text) == 6
        assert token_text == [
            '<s>',
            'Are',
            'Ġthere',
            'Ġthree',
            '?',
            '</s>'
        ]

    def test_read_from_file_question_and_abstract(self):
        reader = QasperReader(context="question_and_abstract")
        instances = ensure_list(reader.read("fixtures/data/qasper_sample_small.json"))
        assert len(instances) == 4

        instance = instances[1]
        assert set(instance.fields.keys()) == {
            "question_with_context",
            "paragraph_indices",
            "global_attention_mask",
            "evidence",
            "answer",
            "metadata",
        }

        token_text = [t.text for t in instance.fields["question_with_context"].tokens]
        assert len(token_text) == 13
        assert token_text == [
            '<s>',
            'Are',
            'Ġthere',
            'Ġthree',
            '?',
            '</s>',
            'This',
            'Ġis',
            'Ġthe',
            'Ġabstract',
            'Ġof',
            'Ġthe',
            'Ġpaper',
        ]

    def test_read_from_file_question_and_introduction(self):
        reader = QasperReader(context="question_and_introduction")
        instances = ensure_list(reader.read("fixtures/data/qasper_sample_small.json"))
        assert len(instances) == 4

        instance = instances[1]
        assert set(instance.fields.keys()) == {
            "question_with_context",
            "paragraph_indices",
            "global_attention_mask",
            "evidence",
            "answer",
            "metadata",
        }

        token_text = [t.text for t in instance.fields["question_with_context"].tokens]
        assert len(token_text) == 15
        assert token_text == [
            '<s>',
            'Are',
            'Ġthere',
            'Ġthree',
            '?',
            '</s>',
            'Introduction',
            '</s>',
            'A',
            'Ġshort',
            'Ġparagraph',
            '</s>',
            'Another',
            'Ġintro',
            'Ġparagraph']

    def test_read_from_file_question_and_evidence(self):
        reader = QasperReader(context="question_and_evidence")
        instances = ensure_list(reader.read("fixtures/data/qasper_sample_small.json"))
        # Yes/No and None answers are ignored.
        assert len(instances) == 2  # instead of 4 for the other readers

        instance = instances[1]
        assert set(instance.fields.keys()) == {
            "question_with_context",
            "paragraph_indices",
            "global_attention_mask",
            "evidence",
            "answer",
            "metadata",
        }

        token_text = [t.text for t in instance.fields["question_with_context"].tokens]
        assert len(token_text) == 13
        assert token_text == [
            '<s>',
            'Is',
            'Ġthis',
            'Ġextract',
            'ive',
            '?',
            '</s>',
            'Method',
            'Ġparagraph',
            'Ġusing',
            'Ġseed',
            'Ġlex',
            'icon',
        ]
