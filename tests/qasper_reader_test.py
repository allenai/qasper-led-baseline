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
            'Ġparagraph']

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

    def test_instance_indexes_correctly(self):
        # Tests the indexed instances against the values input to the LED model in the
        # original qasper_baselines code, accounting for two differences
        # 1. The input_ids need to be pre-padded to window size (1024) in the original
        # longformer code, but the HF code does that internally now.
        # 2. The answer ids need to be shifted right, which the HF code does internally also.
        reader = QasperReader()
        vocabulary = Vocabulary.empty()
        instances = ensure_list(reader.read("fixtures/data/qasper_sample_tiny.json"))
        instance = instances[0]
        instance.index_fields(vocabulary)
        tensor_dict = instance.as_tensor_dict()
        input_ids = tensor_dict['question_with_context']['tokens']['token_ids']
        assert input_ids.tolist() == [
            0, 2264, 16, 5, 5018, 36912, 17505, 116, 2, 46576, 2, 250, 765, 17818,
            2, 21518, 22845, 17818,2, 22816, 6011, 2, 3609, 10679, 17818, 2, 41895,
            7878, 16410, 2, 6323,  5448, 2, 41895,  7878, 16410,  4832, 38304, 6189,
            21528, 42419, 2, 47967, 17818, 634,  5018, 36912, 17505, 2, 48984, 2,
            48984, 17818
        ]
        answer_ids = tensor_dict['answer']['tokens']['token_ids']
        assert answer_ids.tolist() == [0, 102, 32644, 2]
        expected_attention_mask = [
            2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1,
            1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1,
            2, 1, 2, 1, 1
        ]
        assert tensor_dict['global_attention_mask'].tolist() == [x == 2 for x in expected_attention_mask]
