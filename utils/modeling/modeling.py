from typing import Dict, List, Optional, Any, Union, Tuple, Literal
import pandas as pd
from pyspark.sql import DataFrame
from .native_modeling import run_all_anomaly_detectors
from .spark_modeling import run_spark_anomaly_detectors
from ..logger import log


def modeling(
    df: Union[pd.DataFrame, DataFrame],
    methods: List[str],
    eval_metric: str,
    complexity: str,
    use_spark: bool,
) -> Dict[str, Dict[str, Any]]:
    if use_spark:
        reuslt_dict = run_spark_anomaly_detectors(
            df=df, methods=methods, eval_metric=eval_metric, complexity=complexity
        )
        return reuslt_dict
    reuslt_dict = run_all_anomaly_detectors(
        df=df, methods=methods, eval_metric=eval_metric, complexity=complexity
    )
    return reuslt_dict
