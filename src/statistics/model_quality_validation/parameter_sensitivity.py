import numpy as np
from typing import Dict, Any
from .base import ModelEvaluation

class ParameterSensitivity(ModelEvaluation):

    def run(
            self,
            model_result: Dict[str, Any],
            df = None
    ) -> Dict[str, Any]:
        scores = []

        param_scores = model_result.get("parameter_score", None)

        if param_scores:
            scores = list(param_scores.values())
        else:
            scores = [model_result.get("score", 0)]

        if len(scores)>1:
            sensitivity = np.std(scores) / (np.mean(scores)+1e-9)
        else:
            sensitivity = 0.0

        status = "PASS" if sensitivity<0.2 else "WARN"
        suggestion = (
            "Model performance is stable across hyperparameters."
            if status == "PASS"
            else "Model performance varies with hyperparameters."
        )

        return {
            "algorithm": model_result["algorithm"],
            "metric": "Parameter Sensitivity",
            "status": status,
            "score": sensitivity,
            "suggestion": suggestion
        }