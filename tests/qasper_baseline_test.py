from allennlp.common.testing import ModelTestCase
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary

import numpy
from numpy.testing import assert_almost_equal
import torch

#import qasper_baselines.model  # pylint: disable=unused-import
from qasper_baselines.model import QasperBaseline
#import qasper_baselines.dataset_reader  # pylint: disable=unused-import
from qasper_baselines.dataset_reader import QasperReader


class TestQasperBaseline(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            "fixtures/qasper_baseline.jsonnet", "fixtures/data/qasper_sample_small.json"
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_model_gives_correct_logits(self):
        torch.manual_seed(15371)
        reader = QasperReader()
        vocabulary = Vocabulary.empty()
        instances = ensure_list(reader.read("fixtures/data/qasper_sample_tiny.json"))
        instance = instances[0]
        instance.index_fields(vocabulary)
        tensor_dict = instance.as_tensor_dict()
        model = QasperBaseline(vocabulary, 'allenai/led-base-16384')
        question_with_context = {}
        for key, value in tensor_dict['question_with_context'].items():
            if key == "tokens":
                question_with_context[key] = {"token_ids": value["token_ids"].unsqueeze(0)}
            else:
                question_with_context[key] = value.unsqueeze(0)
        answer = {"tokens": {"token_ids": tensor_dict['answer']["tokens"]["token_ids"].unsqueeze(0)}}
        model.eval()
        model_outputs = model(question_with_context=question_with_context,
                              paragraph_indices=tensor_dict['paragraph_indices'].unsqueeze(0),
                              global_attention_mask=tensor_dict['global_attention_mask'].unsqueeze(0),
                              answer=answer,
                              metadata=[tensor_dict['metadata']])
        model_outputs['answer_logits'].size() == [1, 4, 50265]
        assert_almost_equal(
            model_outputs['loss'].detach().numpy(),
            numpy.asarray(7.8690)
        )
        assert_almost_equal(
            model_outputs['answer_logits'][:, :, :5].detach().numpy(),
            numpy.asarray(
                [[[32.0200,  6.0771, 16.3266,  5.6827, 12.0250],
                  [ 6.4264, -1.3621, 18.6183,  0.4633, 11.6951],
                  [-5.7808, -4.2320, 13.8344, -3.5288, 11.2021],
                  [-1.7276, -4.3160, 12.7081, -4.2964, 8.3145]]]
            )
        )
        assert model_outputs['predicted_answers'] != ['']
