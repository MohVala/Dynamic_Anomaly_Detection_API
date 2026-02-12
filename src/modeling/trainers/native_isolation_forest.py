import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import IsolationForest

from ...statistics.metrics.clustering_score import clustering_score
from ...utils.logger import log

def train_native_isolation_forest (
        df: pd.DataFrame,
        eval_metric: str,
        complexity: str
) -> Dict[str, Dict[str, Any]]:
    log("modeling", "start", f"model is Native Isolation Forest and evaluation is {eval_metric}")
    
    data = df.loc[:, df.dtypes != "object"]

    param_grids = {
        "n_estimator": (
            range(10,20,10) if complexity == "s" else range(10,100, 10)
        ),
        "max_samples": (
            np.arange(0.1, 0.4, 0.1) if complexity == "s" else np.arange(0.1,1, 0.1)
        ),
        "contamination": (
            np.linspace(0.01, 0.5, 2) if complexity == "s" else np.linspace(0.01, 0.5, 10)
        ),
        "max_features":
        np.arange(0.1,0.3, 0.1) if complexity == "s" else np.arange(0.1, 1, 0.1)
    }
    
    best_score = -1
    best_model = None
    best_params = None
    best_preds = None

    for param in ParameterGrid(param_grids):
        model = IsolationForest(**param)
        preds = model.fit_predict(data)
        score = clustering_score(
            data = data,
            labels= preds,
            metric= eval_metric
        )

        if eval_metric != "davies_bouldin":
            if best_score is None or score>best_score:
                best_score = score
                best_model = model
                best_params = param
                best_preds = preds
        else:
            if best_score is None or score<best_score:
                best_score = score
                best_model = model
                best_params = param
                best_preds = preds
        
    log("modeling", "end", f"model is Native Isolation Forest and evaluation is {eval_metric} and score is {best_score}")

    return {
            "algorithm": "Isolation_Forest",
            "model": best_model,
            "score": best_score,
            "parameters": best_params,
            "predictions": best_preds,
            "raw_labels": best_preds
    }
