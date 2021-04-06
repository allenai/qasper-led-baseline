from allennlp.common.testing import ModelTestCase

import qasper_baselines.model  # pylint: disable=unused-import
import qasper_baselines.dataset_reader  # pylint: disable=unused-import


class TestQasperBaseline(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            "fixtures/qasper_baseline.jsonnet", "fixtures/data/qasper_sample_small.json"
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
