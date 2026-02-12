from typing import Dict, List, Optional, Any, Union, Tuple, Literal
from pyspark.sql import SparkSession
from pyspark import SparkConf
from .logger import log


# Spark initialization Function:
def init_spark(app_name: str = "AnomalyDetection") -> SparkSession:
    conf = SparkConf().setAppName(app_name).setMaster("local[*]")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    log("spark", "init", f"Spark session initialized with app_name={app_name}")
    return spark
