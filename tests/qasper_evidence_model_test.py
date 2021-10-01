from allennlp.common.testing import ModelTestCase
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary

import numpy
from numpy.testing import assert_almost_equal
import torch

import qasper_baselines.evidence_retrieval_model  # pylint: disable=unused-import
import qasper_baselines.evidence_retrieval_reader  # pylint: disable=unused-import


class TestQasperBaseline(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            "fixtures/qasper_evidence_dpr.jsonnet", "fixtures/data/qasper_sample_small.json"
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
