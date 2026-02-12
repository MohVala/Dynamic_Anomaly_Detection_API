import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans

from ...statistics.metrics.clustering_score import clustering_score
from ...utils.logger import log
def train_native_kmeans(
        df: pd.DataFrame,
        eval_metric: str,
        complexity: str,
) -> Dict[str, Dict[str,Any]]:

    log("modeling", "start", f"model is Native KMeans and evaluation is {eval_metric}")
    
    data = df.loc[:, df.dtypes!="object"]


    param_grid = {
        "n_clusters": range(3,5,1) if complexity == "s" else range(3,10,1)
    }

    best_score = -1
    best_model = None
    best_params = None
    best_pred = None

    for params in ParameterGrid(param_grid):
        
        model = KMeans(**params, random_state=42)
        preds = model.fit(data)
        score = clustering_score(
            data=data,
            labels= preds,
            metric=eval_metric
            )
        
        if eval_metric != "davies_bouldin":
            if best_score is None or score>best_score:
                best_score = score
                best_model = model
                best_params = param_grid
                best_pred = preds
        else:
            if best_score is None or score<best_score:
                best_score = score
                best_model = model
                best_params = param_grid
                best_pred = preds

    log("modeling", "end", f"modelling is Native KMeans and evaluation is {eval_metric} and score is  {best_score}")
    
    return {
            "algorithm": "KMeans",
            "model": best_model,
            "score": best_score,
            "parameters": best_params,
            "predictions": best_pred,
            "raw_labels": best_pred
        }