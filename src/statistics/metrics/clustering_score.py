import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from pyspark.ml.evaluation import ClusteringEvaluator

def clustering_score(
        data: np.ndarray,
        labels: np.ndarray,
        metric: str
) -> float:
    if len(set(labels)) <=1:
        return -1
    
    if metric == "silhouette":
        return silhouette_score(data, labels)
    
    if metric == "davies_bouldin":
        return davies_bouldin_score(data, labels)
    
    if metric == "calinski_harabasz":
        return calinski_harabasz_score(data, labels)
    
    raise ValueError(f"Unsupported metric {metric}")