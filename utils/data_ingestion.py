from typing import Dict, List, Optional, Any, Union, Tuple, Literal
import pandas as pd
import requests
from pyspark.sql import SparkSession
from .spark_utils import init_spark
from .logger import log
def ingest_api_to_dataframe(api_url: str)-> pd.DataFrame:
    def flatten_json(y: Union[Dict[str, Any], List[Any]])-> Dict[str, Any]:
        out = {}

        def flatten(x: str, name: str ='')->None:
            if isinstance(x, dict):
                for a in x:
                    flatten(x[a], f'{name}{a}_')
            elif isinstance(x,list):
                for i, a in enumerate(x):
                    flatten(a,f'{name}{i}_')
            else:
                out[name[:-1]] = x
            
        flatten(y)
        return out
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        raw_json: Union[Dict[str, Any], List[Any]] = response.json()

        # handle list or single object:
        if isinstance(raw_json, list):
            data: List[Dict[str, Any]] = [flatten_json(item) for item in raw_json]
        else:
            data = [flatten_json(raw_json)]
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f" Error Happened ******** {e}***********")
        return pd.DataFrame()

def ingest_data(api_url:str, use_spark:bool)->Union[pd.DataFrame,"pyspark.sql.DataFrame"]:
    df = ingest_api_to_dataframe(api_url)
    if use_spark:
        spark = init_spark()
        spark_df = spark.createDataFrame(df)
        log("data_ingestion", "spark", f"Ingested data as Spark DataFrame with {spark_df.count()} rows")
        return spark_df
    return df

