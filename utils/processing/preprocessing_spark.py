from typing import Dict, List, Optional, Any, Union, Tuple, Literal
from pyspark.sql import DataFrame
from pyspark.sql.functions import when,col
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.clustering import KMeans

# duplicates handling:
def duplications_spark(df: DataFrame)->DataFrame:
    return df.dropDuplicates()

def fill_missing_spark_kmeans(
        df:DataFrame, 
        n_cluster:int=5
        )->DataFrame:
    # select numeric columns:
    numeric_cols = [
        f.name for f in df.schema.fields
        if f.dataType.simpleString() in ("int", "double","flaot","bigint")
    ]
    if not numeric_cols:
        return df
    
    # keep non-null values records:
    df_train = df.dropna(subset=numeric_cols)

    if df_train.rdd.isEmpty():
        return df
    
    # assemble features:
    assembler = VectorAssembler(
        inputCols=numeric_cols,
        outputCol="feature"
    )

    df_train_vec = assembler.transform(df_train)

    # scale features:
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True
    )

    scaler_model = scaler.fit(df_train_vec)
    df_train_scaled = scaler_model.transform(df_train_vec)

    # train KMeans:
    kmeans = KMeans(
        k = n_cluster,
        seed=42,
        featuresCol="scaled_features",
        predictionCol="cluster"
    )
    kmeans_model = kmeans.fit(df_train_scaled)

    # assign clusters to all rows (fill nulls with means)
    means = {
        c: df_train.selectExpr(f"avg({c})".first()[0])
        for c in numeric_cols
    }

    df_filled_temp = df
    for c, m in means.items():
        df_filled_temp = df_filled_temp.withColumn(
            c, when(col(c).isNull(),m).otherwise(col(c))
        )
    df_all_vec = assembler.transform(df_filled_temp)
    df_all_scaled = scaler_model.transform(df_all_vec)

    # extract cluster centers
    centers = kmeans_model.clusterCenters()

    # replace missing values using cluster centers
    for i, c in enumerate(numeric_cols):
        df_clustered = df_clustered.withColumn(
            c,
            when(
                col(c).isNull(),
                col("cluster").cast("int").apply(
                lambda k: float(centers[k][i])
                )
            ).otherwise(col(c))
        )
    return df_clustered.drop(
        "features",
        "scaled_features",
        "cluster"
    )

def normalization_features_spark(df: DataFrame) -> DataFrame:

    # 1️⃣ Detect numeric columns
    numeric_cols = [
        f.name for f in df.schema.fields
        if f.dataType.simpleString() in ("int", "double", "float", "bigint")
    ]

    if not numeric_cols:
        return df

    # 2️⃣ Assemble numeric features
    assembler = VectorAssembler(
        inputCols=numeric_cols,
        outputCol="features"
    )

    df_vec = assembler.transform(df)

    # 3️⃣ Min-Max scaling
    scaler = MinMaxScaler(
        inputCol="features",
        outputCol="scaled_features"
    )

    scaler_model = scaler.fit(df_vec)
    df_scaled = scaler_model.transform(df_vec)

    # 4️⃣ Convert vector back to columns
    df_array = df_scaled.withColumn(
        "scaled_array",
        vector_to_array(col("scaled_features"))
    )

    for i, c in enumerate(numeric_cols):
        df_array = df_array.withColumn(
            c,
            col("scaled_array")[i]
        )

    # 5️⃣ Cleanup
    return df_array.drop(
        "features",
        "scaled_features",
        "scaled_array"
    )

