from .native_preprocessing import (
    duplications,
    fill_missing_with_kmeans,
    normalization_features,
)
from .preprocessing_spark import (
    duplications_spark,
    fill_missing_spark_kmeans,
    normalization_features_spark,
)
from typing import Union
from pyspark.sql import DataFrame
import pandas as pd


def process_data(
    df: Union[pd.DataFrame, DataFrame], use_spark: bool
) -> Union[pd.DataFrame, DataFrame]:
    if use_spark:
        df = duplications_spark(df)
        df = fill_missing_spark_kmeans(df)
        df = normalization_features_spark(df)
        return df
    df = duplications(df)
    df = fill_missing_with_kmeans(df)
    df = normalization_features(df)
    return df
