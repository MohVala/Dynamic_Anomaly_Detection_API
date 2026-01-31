from typing import Dict, List, Any
from pyspark.sql import DataFrame
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator
from ..logger import log


def run_spark_anomaly_detectors(
    df: DataFrame, methods: List[str], eval_metric: str, complexity: str
) -> Dict[str, Dict[str, Any]]:

    result: Dict[str, Dict[str, Any]] = {}

    numeric_cols = [c for c, t in df.dtypes if t != "string"]
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    df_vec = assembler.transform(df)

    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")

    for method in methods:
        if method == "kmeans":

            log("modeling", "start", "Kmeans(sparl)")

            k_range = range(3, 5) if complexity == "s" else range(3, 10)

            best_score, best_model = -1, None

            for k in k_range:
                model = KMeans(k=k, seed=42)
                fitted = model.fit(df_vec)
                score = evaluator.evaluate(fitted.transform(df_vec))

                if score > best_score:
                    best_score = score
                    best_model = fitted

                result[method] = {
                    "algorithm": "spark KMeans",
                    "model": best_model,
                    "score": best_score,
                }
            log("modeling", "end", "KMeans(spark)")

        elif method == "bisecting_kmeans":
            log("modeling", "start", "bisecting_kmeans(spark)")

            k_range = range(3, 5) if complexity == "s" else range(3, 10)

            best_score, best_model = -1, None

            for k in k_range:
                model = BisectingKMeans(k=k, seed=52)
                fitted = model.fit(df_vec)
                score = evaluator.evaluate(fitted.transform(df_vec))

            result[method] = {
                "algorithm": "Spark BisectingKMeans",
                "model": fitted,
                "score": score,
            }

            log("modeling", "end", "bisecting_kmeans(spark)")
