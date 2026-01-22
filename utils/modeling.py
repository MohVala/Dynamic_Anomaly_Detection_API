import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Any, Union, Tuple, Literal

from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from .logger import log



def get_time_series_split (X: np.ndarray | pd.DataFrame, n_splits: int = 5) -> List[Tuple[np.ndarray,np.ndarray]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return [(train_index, test_index) for train_index, test_index in tscv.split(X)]


def calculate_clustering_score(
        data: np.ndarray| pd.DataFrame, 
        labels:np.ndarray| pd.Series, 
        metric: str
        )->float:
    if len(set(labels))<=1:
        return -1
    if metric == "silhouette":
        return silhouette_score(data, labels)
    if metric == "davies_bouldin":
        return davies_bouldin_score(data, labels)
    if metric == "calinski_harabasz":
        return calinski_harabasz_score(data, labels)
    else:
        return ValueError(f"Unsupported evaluation metric: {metric}")

def run_all_anomaly_detectors(
        df: pd.DataFrame, 
        methods:List[str],
        eval_metric:str,
        complexity: str
 ) -> Dict[str,Dict[str, Any]]:
    
    result: Dict[str, Dict[str, Any]] = {}
    
    data = df.loc[:, df.dtypes !='object']

    for method in methods:
        if method == 'isolation_forest':
            log("simple_modeling_1/3", "start", "simple isolation forest modeling ")
            print(f"start of modeling: {method}")
            param_grid = {
                'n_estimators': range(10,20, 10) if complexity=='s' else range(10,100, 10),
                'max_samples': np.arange(0.1,0.4, 0.1) if complexity=='s' else np.arange(0.1,1, 0.1),
                'contamination': np.linspace(0.01, 0.5, 2) if complexity=='s' else np.linspace(0.01, 0.5, 10),
                'max_features': np.arange(0.1, 0.3, 0.1) if complexity=='s' else np.arange(0.1, 1, 0.1)
            }

            best_score = -1
            best_model = None
            best_params = None
            best_preds = None

            for params in ParameterGrid(param_grid):
                model = IsolationForest(**params, random_state=42)
                preds = model.fit_predict(data)
                score = calculate_clustering_score(data = data,labels= preds, metric=eval_metric)
                if score>best_score:
                    best_score = score
                    best_model = model
                    best_params = params
                    best_preds = np.where(preds==-1,1,0)
            result[method] = {
                'algorithm': "Isolation Forest",
                'model' : best_model,
                'score' : best_score,
                'parameters': best_params,
                'anomaly_detection': best_preds
            }
            print(f"end of modeling: {method} \n ====== \n")
            log("simple_modeling_1/3", "end", "simple isolation forest modeling ")


        elif method == 'kmeans':
            log("simple_modeling_2/3", "start", "simple kmeans modeling ")
            print(f"start of modeling: {method}")

            param_grid = {
                'n_clusters': range(3,5, 1) if complexity=="s" else range(3,10, 1)
            }

            best_score = -1
            best_model = None
            best_params = None
            best_preds = None

            for params in ParameterGrid(param_grid):
                model = KMeans(**params, random_state=42)
                preds = model.fit_predict(data)
                score = calculate_clustering_score(data = data,labels= preds, metric=eval_metric)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = params
                    best_preds = preds
            result[method] = {
                'algorithm': "KMeans",
                'model' : best_model,
                'score' : best_score,
                'parameters': best_params,
                'anomaly_detection': best_preds
            }

            print(f"end of modeling: {method} \n ====== \n")
            log("simple_modeling_2/3", "end", "simple kmeans modeling ")
        elif method == 'dbscan':
            log("simple_modeling_3/3", "start", "simple dbscan modeling ")
            print(f"start of modeling: {method}")

            param_grid = {
                'eps' : np.arange(0.1,0.2, 0.1) if complexity=="s" else np.arange(0.1,1, 0.1),
                'min_samples': range(3, 7, 2) if complexity=="s" else range(3, 21, 2)
            }

            best_score = -1
            best_model = None
            best_params = None
            best_preds = None

            for params in ParameterGrid(param_grid):
                model = DBSCAN(**params)
                preds = model.fit_predict(data)
                score = calculate_clustering_score(data = data,labels= preds, metric=eval_metric)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = params
                    best_preds = preds
            result[method] = {
                'algorithm': "DBSCAN",
                'model' : best_model,
                'score' : best_score,
                'parameters': best_params,
                'anomaly_detection': best_preds
            }
            print(f"end of modeling: {method} \n ====== \n")
            log("simple_modeling_3/3", "end", "simple dbscan modeling ")
    return result
