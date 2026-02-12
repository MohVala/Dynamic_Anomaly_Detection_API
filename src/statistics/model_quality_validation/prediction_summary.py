from typing import Dict, Any
import numpy as np
from .base import ModelEvaluation

class PredictionSummary(ModelEvaluation):

    def run(
            self,
            model_result:Dict[str, Any],
            df = None
    ) -> Dict[str, Any]:
        preds = model_result.get("predictions", None)
        if preds is None:
            preds = model_result.get("raw_labels", [])
        
        preds = np.array(preds)
        total = len(preds)

        unique, count = np.unique(preds, return_count = True)
        distribution = dict(zip(unique.tolist(), count.tolist()))

        status = "PASS"
        suggestion = "Prediction distribution summarized for reporting."

        return {
            "algorithm": model_result["algorithm"],
            "metric": "Prediction Summary",
            "status": status,
            "score": None,
            "suggestion": suggestion,
            "distribution": distribution,
            "total_predictions": total
        }