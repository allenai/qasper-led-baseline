import json
from overrides import overrides

from allennlp.common import JsonDict
from allennlp.predictors import Predictor

@Predictor.register('qasper')
class QasperPredictor(Predictor):
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        return json.dumps({"question_id": outputs["question_id"],
                           "predicted_answer": outputs["predicted_answers"],
                           "predicted_evidence": outputs["predicted_evidence"]}) + "\n"
