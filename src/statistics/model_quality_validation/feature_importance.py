from typing import Dict, Any
import numpy as np
from .base import ModelEvaluation

class FeaturesImportance(ModelEvaluation):

    def run(
            self,
            model_result: Dict[str, Any],
            df = None
    ) -> Dict[str, Any]:
        model = model_result.get("model", None)
        if hasattr(model, "feature_importance_"):
            importances = model.feature_importance_
            avg_importance = np.mean(importances)
            status = "PASS" if avg_importance >0.01 else "WARN"
            suggestion = (
                "Features contribute sufficiently to model"
                if status == "PASS"
                else "Most Features have very low importance, consider adding new features (column) into data source."
            )

        else:
            avg_importance = None
            status = "PASS"
            suggestion = "Feature importance not available for this model(s) types."
        return {
            "algorithm": model_result["algorithm"],
            "metric": "Feature Contribution",
            "status": status,
            "score": avg_importance,
            "suggestion": suggestion,
        }