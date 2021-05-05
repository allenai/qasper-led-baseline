from overrides import overrides

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor

@Predictor.register('qasper')
class QasperPredictor(Predictor):
    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        self._dataset_reader.apply_token_indexers(instance)
        output = self._model.forward_on_instance(instance)
        return sanitize(outputs)
