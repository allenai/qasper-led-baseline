# pylint: disable=no-self-use,invalid-name
from allennlp.common.util import ensure_list

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
        assert len(token_text) == 29
        assert token_text[:15] == [
            "<s>",
            "Are",
            "Ġthere",
            "Ġthree",
            "?",
            "</s>",
            "</s>",
            "A",
            "Ġshort",
            "Ġparagraph",
            "</s>",  # default paragraph separator
            "Another",
            "Ġintro",
            "Ġparagraph",
            "</s>",
        ]

        assert len(instance["paragraph_indices"]) == 5
        # This is the first token after the separator for each paragraph.
        assert [x.sequence_index for x in instance["paragraph_indices"]] == [
            7,
            11,
            15,
            19,
            26,
        ]

        expected_mask = [False] * 29
        # question tokens and paragraph start indices
        for x in [0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19, 26]:
            expected_mask[x] = True

        assert instance["global_attention_mask"].tensor.tolist() == expected_mask

        assert len(instance["evidence"]) == len(instance["paragraph_indices"])
        assert instance["evidence"].tensor.tolist() == [0, 0, 0, 1, 0]

        answer_text = [t.text for t in instance.fields["answer"].tokens]
        assert answer_text == ["Yes"]

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
        assert answer_text == ["a", "Ġvocabulary"]

        answer_text = [t.text for t in instances[2].fields["answer"].tokens]
        assert answer_text == ["Un", "answer", "able"]

        answer_text = [t.text for t in instances[3].fields["answer"].tokens]
        assert answer_text == ["Conclusion", "Ġparagraph"]
