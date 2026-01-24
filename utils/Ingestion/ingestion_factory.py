
from typing import Union
import pandas as pd
from pyspark.sql import DataFrame

from .native_ingestion import ingest_api_to_dataframe
from .spark_ingestion import ingest_api_to_spark

def ingest_data(api_url:str, use_spark:bool)->Union[pd.DataFrame,DataFrame]:
    if use_spark:
        return ingest_api_to_spark(api_url)
    return ingest_api_to_dataframe(api_url)