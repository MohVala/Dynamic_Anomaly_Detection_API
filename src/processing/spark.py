from typing import Dict, List, Optional, Any, Union, Tuple, Literal
from pyspark.sql import DataFrame, Row
from pyspark.sql.functions import when, col, lit, avg
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.clustering import KMeans


# duplicates handling:
def duplications_spark(df: DataFrame) -> DataFrame:
    return df.dropDuplicates()


def fill_missing_spark_kmeans(df: DataFrame, n_cluster: int = 5) -> DataFrame:
    # select numeric columns:
    numeric_cols = [
        f.name
        for f in df.schema.fields
        if f.dataType.simpleString() in ("int", "double", "float", "bigint")
    ]
    if not numeric_cols:
        return df
    # 2️⃣ Compute global means (single Spark action)
    means = df.select([avg(c).alias(c) for c in numeric_cols]).first().asDict()

    # 3️⃣ Fill missing values
    df_filled = df
    for c, m in means.items():
        df_filled = df_filled.withColumn(
            c, when(col(c).isNull(), lit(m)).otherwise(col(c))
        )

    return df_filled


def normalization_features_spark(df: DataFrame) -> DataFrame:

    # 1️⃣ Detect numeric columns
    numeric_cols = [
        f.name
        for f in df.schema.fields
        if f.dataType.simpleString() in ("int", "double", "float", "bigint")
    ]

    if not numeric_cols:
        return df

    # 2️⃣ Assemble numeric features
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")

    df_vec = assembler.transform(df)

    # 3️⃣ Min-Max scaling
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")

    scaler_model = scaler.fit(df_vec)
    df_scaled = scaler_model.transform(df_vec)

    # 4️⃣ Convert vector back to columns
    df_array = df_scaled.withColumn(
        "scaled_array", vector_to_array(col("scaled_features"))
    )

    for i, c in enumerate(numeric_cols):
        df_array = df_array.withColumn(c, col("scaled_array")[i])

    # 5️⃣ Cleanup
    return df_array.drop("features", "scaled_features", "scaled_array")
