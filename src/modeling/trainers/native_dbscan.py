import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.model_selection import ParameterGrid
from sklearn.cluster import DBSCAN

from ...statistics.metrics.clustering_score import clustering_score
from ...utils.logger import log

def train_native_dbscan(
        df: pd.DataFrame,
        eval_metric: str,
        complexity: str
) -> Dict[str, Dict[str, Any]]:
    log("modeling", "start", f"model is Native DBSCAN and evaluation is {eval_metric}")
    
    data = df.loc[:, df.dtypes != "object"]


    param_grid = {
        "eps": (
            np.arange(0.1,0.2,0.1) if complexity == "s" else np.arange(0.1,1,0.1)
            ),
        "min_samples": (
            range(3,7,2) if complexity == "s" else range(3,21,2)
            )
    }
    best_score = -1
    best_model = None
    best_params = None
    best_pred = None

    for params in param_grid:
        model = DBSCAN(**params)
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
                best_params = params
                best_pred = preds
        else:
            if best_score is None or score<best_score:
                best_score = score
                best_model = model
                best_params = params
                best_pred = preds
        
    log("modeling", "end", f"model is Native DBSCAN and evaluation is {eval_metric} and score is {best_score}")

    return {
            "algorithm": "DBSCAN",
            "model": best_model,
            "score": best_score,
            "parameters": best_params,
            "predictions": best_pred,
            "raw_labels": best_pred
        }

             
