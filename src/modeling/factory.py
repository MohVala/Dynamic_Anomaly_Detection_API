from typing import Dict, Any, Union, List
import pandas as pd
from pyspark.sql import DataFrame

from .trainers import (
    native_dbscan,
    native_isolation_forest,
    native_kmeans
)
# from .trainers.spark import run_spark_anomaly_detectors
from ..utils.logger import log


# def modeling(
#     df: Union[pd.DataFrame, DataFrame],
#     methods: List[str],
#     eval_metric: str,
#     complexity: str,
#     use_spark: bool,
# ) -> Dict[str, Dict[str, Any]]:
#     if use_spark:
#         reuslt_dict = run_spark_anomaly_detectors(
#             df=df, methods=methods, eval_metric=eval_metric, complexity=complexity
#         )
#         return reuslt_dict
#     reuslt_dict = run_all_anomaly_detectors(
#         df=df, methods=methods, eval_metric=eval_metric, complexity=complexity
#     )
#     return reuslt_dict


class ModelFactory:
    def __init__(
            self,
            df: pd.DataFrame, # Union[pd.DataFrame, DataFrame],
            methods: list[str],
            eval_metric: str,
            complexity: str,
            use_spark: str
            ):
        self.df = df
        self.methods = methods
        self.eval_metric = eval_metric
        self.complexity = complexity
        self.use_spark = use_spark

    def run_all_model(self) -> Dict[str, Dict[str, Any]]:
        results = {}

        for method in self.methods:
            if method == "isolation_forest":
                results[method] = native_isolation_forest(
                    df = self.df,
                    eval_metric = self.eval_metric,
                    complexity = self.complexity
                )
            elif method == "kmeans":
                results[method] = native_kmeans(
                    df = self.df,
                    eval_metric = self.eval_metric,
                    complexity = self.complexity
                )
            elif method == "dbscan":
                results[method] = native_kmeans(
                    df = self.df,
                    eval_metric = self.eval_metric,
                    complexity = self.complexity
                )

            else:
                print("Warning, The Model names are not valid or not available.")

        return results