# -------------------------------
# Configuration and Libraries
# -------------------------------
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

from utils.logger import log, log_stream

log("import", "start", "Importing necessary libraries")

from typing import List

import pandas as pd

from utils.data_ingestion import ingest_data
from utils.preprocessing import (
    duplications,
    fill_missing_with_kmeans,
    normalization_features
)
from utils.modeling import run_all_anomaly_detectors
from utils.report_generator import generate_html_report

log("import", "end", "Importing necessary libraries")

# ----------------------------
# Data Ingestion:
# ----------------------------
log("data_ingestion", "start", "data ingestion from source")

API_URL = config["run"]["api_url"]
use_spark: bool = config['run'].get('use_spark')
ingested_df = ingest_data(api_url=API_URL, use_spark=use_spark)
if use_spark:
    print(f"✅ Spark DataFrame ingested with {ingested_df.count()} rows")
else:
    print(f"✅ Pandas DataFrame ingested with {ingested_df.shape[0]} rows")
log("data_ingestion", "end", "data ingestion from source")

visualization_question = "y" if config["run"]["visualization"] else "n"

# Just for test:
numeric_cols = [
    'passenger_count', 'trip_distance', 'ratecodeid', 'pulocationid', 'dolocationid',
    'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
    'improvement_surcharge', 'total_amount'
]

for col in numeric_cols:
    ingested_df[col] = pd.to_numeric(ingested_df[col], errors='coerce')  # converts and sets invalid parsing to NaN

log("data_ingestion", "start", "data ingestion from source")

# ----------------------------
# Data Preparation:
# ---------------------------- 

log("data_preparation", "start", "data preparation")

# remove duplicates and keep fisrt one only:
duplications(ingested_df)

filled_df = fill_missing_with_kmeans(ingested_df, n_cluster=5)

normed_df = normalization_features(filled_df)

log("data_preparation", "end", "data preparation")

# ----------------------------
# Modeling:
# ---------------------------- 

# calculate and choose clustering evaluation metrics:
eval_metric = config["evaluation"]["primary_metric"]
# methods from config file:
methods = config["modeling"]["models"]
# model complexity
model_complexity  = "s" if config['modeling']['mode']=='simple' else "c"

result_dict = run_all_anomaly_detectors(
    df=normed_df,
    methods=methods,
    eval_metric=eval_metric,
    complexity=model_complexity
)

# ----------------------------
# Reporting:
# ---------------------------- 

generate_html_report(
    api_url=API_URL,
    df=ingested_df,
    normed_df=normed_df,
    result_dict=result_dict,
    logs=log_stream.getvalue()
)