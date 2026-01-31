from typing import Dict, Any, List, Union
import requests
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from ..spark_utils import init_spark
from utils.logger import log
from .api_flatten import flatten_json


def ingest_api_to_spark(api_url: str) -> DataFrame:

    spark = init_spark()

    try:
        log("data_ingestion", "start", "Ingesting API data into Spark DataFrame")
        response = requests.get(api_url)
        response.raise_for_status()
        raw_json: Union[Dict[str, Any], List[Any]] = response.json()
        # handle list or single object:
        if isinstance(raw_json, list):
            data: List[Dict[str, Any]] = [flatten_json(item) for item in raw_json]
        else:
            data = [flatten_json(raw_json)]

        if not data:
            log("data_ingestion", "spark", "No records received from API")
            return spark.createDataFrame([], StructType([]))

        # Build schema dynamically (all strings at ingestion stage)
        schema = StructType(
            [StructField(col, StringType(), nullable=True) for col in data[0].keys()]
        )

        spark_df = spark.createDataFrame(data, schema=schema)

        log(
            "data_ingestion",
            "spark",
            f"Ingested Spark DataFrame with {spark_df.count()} rows",
        )

        return spark_df
    except Exception as e:
        log("data_ingestion", "error", str(e))
        return spark.createDataFrame([], StructType([]))
