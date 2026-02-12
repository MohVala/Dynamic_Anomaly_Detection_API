import numpy as np
import pandas as pd
from typing import Dict, Any
from .base import ModelEvaluation

class RepeatedSubmodeling(ModelEvaluation):
    """Evaluates stability of model prediction across subsamples"""

    def __init__(
            self,
            n_iter: int = 5,
            sample_frac: float = 0.8
            ):
        self.n_iter = n_iter
        self.sample_frac = sample_frac

    
    def run(
            self,
            model_result: Dict[str,Any],
            df: pd.DataFrame, # Union[pd.DataFrame DataFrame]
    ) -> Dict[str, Any]:
        label_list = []
        for _ in range(self.n_iter):
            sample_df = df.sample(frac=self.sample_frac, random_state=np.random.randint(1000))
            model = model_result["model"]
            try:
                preds = model.fit_predict(sample_df)
            except Exception:
                preds = model.predict(sample_df)

            label_list.append(preds)

        # calculate stability as average pairwise Jaccard similarity

        stablity_score = []

        for i in range(len(label_list)):
            for j in range(i+1, len(label_list)):
                a = label_list[i]
                b = label_list[j]
                intersection = np.sum(a==b)
                stability = intersection/len(a)
                stablity_score.append(stability)
        avg_stability = np.mean(stablity_score)

        status = "PASS" if avg_stability > 0.8 else "WARN"

        suggestion = (
            "Model is stable across subsamples."
            if status == "Pass"
            else "Model predictions vary across subsamples."
        )

        return {
            "algorithm": model_result["algorithm"],
            "metric": "Repeated Subsampling Stabality",
            "status": status,
            "score": avg_stability,
            "suggestion": suggestion
        }