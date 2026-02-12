from utils.logger import log, log_stream
from src.ingestion.factory import ingest_data
from src.processing.factory import process_data
from src.modeling.factory import ModelFactory
from src.modeling.model_selection import ModelSelection
from reporting.report_generator import generate_html_report
from core.exceptions import (
    IngestionError,
    ProcessingError,
    ModellingError,
    ReportingError,
    ValidationError
)
from ..statistics.data_quality_validation.column_check import ColumnCountCheck
from ..statistics.data_quality_validation.volumn_check import VolumeCheck
from ..statistics.data_quality_validation.null_ratio import NullRatioCheck
from ..statistics.data_quality_validation.row_duplicate_check import RowDuplicateCheck
from ..statistics.data_quality_validation.column_duplicare_check import ColumnDuplicateCheck

from ..statistics.features_quality_validation.duplication_check import DupicatedCheck
from ..statistics.features_quality_validation.missing_check import MissingCheck
from ..statistics.features_quality_validation.normalization_check import NormalizationCheck
from ..statistics.features_quality_validation.correlation_check import CorrelationCheck
from ..statistics.features_quality_validation.distribution_sanity_check import DistributionSanityCheck

from ..statistics.model_quality_validation.feature_importance import FeaturesImportance
from ..statistics.model_quality_validation.parameter_sensitivity import ParameterSensitivity
from ..statistics.model_quality_validation.prediction_summary import PredictionSummary
from ..statistics.model_quality_validation.repeated_subsampling import RepeatedSubmodeling

from ..statistics.decision_engine import DecisionEngine

import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType


def run_pipeline(context):
    """
    Full Dynamic Anomaly Detection Pipeline:
    1. Data Ingestion
    2. Data Preprocessing
    3. Modelling
    4. Reporting
    """

    # ----------------------------
    # Data Ingestion:
    # ----------------------------
    log("Data Ingestion", "start", "ingestion from source")

    try:
        df = ingest_data(api_url=context.api_url)
        if context.use_spark:
            print(f"Spark DataFrame ingested with {df.count()} rows.")
        else:
            print(f"Pandas DataFrame ingested with {df.shape[0]} rows.")
    except Exception as e:
        raise IngestionError(f"Data ingestion failed: {e}")
    
    log("Data Ingestion", "end", "data ingestion completed")
    # ----------------------------
    # Statistics: Data Quality and Validation
    # ----------------------------
    log("statistics: Data Quality", "start", "ColumnCountCheck, VolumeCheck, NullRatioCheck, RowDuplicateCheck, ColumnCountCheck")

    checks = [
        ColumnCountCheck(context),
        VolumeCheck(context),
        NullRatioCheck(context),
        RowDuplicateCheck(context),
        ColumnCountCheck(context),
    ]
    decision_result = [check.run(df) for check in checks]
    decision = DecisionEngine().decide(decision_result)

    if decision["final_status"] == "FAIL":
        raise ValidationError(
            f"Data Quality and validation failed: {decision["details"]}"
            )
    log("statistics: Data Quality", "end", "ColumnCountCheck, VolumeCheck, NullRatioCheck, RowDuplicateCheck, ColumnCountCheck")
    
    # ----------------------------
    # Data Preparation
    # ----------------------------
    log("Data Preprocessing", "start", "start of Processing data")

    # example of cast numeric columns:
    numeric_cols = [
    "passenger_count",
    "trip_distance",
    "ratecodeid",
    "pulocationid",
    "dolocationid",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    ]

    if context.use_spark:
        for c in numeric_cols:
            df = df.withColumn(c, col(c).cast(DoubleType()))
    else:
        for c in numeric_cols:
            df = pd.to_numeric(df[c], errors="coerce")
    
    try:
        df_preprocessed = process_data(df=df, use_spark=context.use_spark)
    except Exception as e:
        raise ProcessingError(f"Data Processing failed: {e}")
    
    log("Data Preprocessing", "end", "Processing completed")
    # ----------------------------
    # Statistics: Features Quality and Validation
    # ----------------------------
    log("Statistics: Features Quality", "start", "")

    checks_features = [
        DupicatedCheck(context),
        MissingCheck(context),
        NormalizationCheck(context),
        CorrelationCheck(context),
        DistributionSanityCheck(context)
    ]
    decision_result_features = [check.run(df_preprocessed) for check in checks_features]
    decision = DecisionEngine().decide(decision_result_features)

    if decision["final_status"] == "FAIL":
        raise ValidationError(
            f"Features are not valid statistically, details: {decision['details']}"
        )
    log("Statistics: Features Quality", "end", "")
    
    # ----------------------------
    # Modeling
    # ----------------------------

    log("Modelling","start", "All modeling started")

    methods = context.config['modeling']['models']
    eval_metric = context.config['evaluation']['primary_metric']
    complexity = "s" if context.config['modeling']['mode'] == 'simple' else "c"

    try:
        factory  = ModelFactory(
            df = df_preprocessed,
            methods=methods,
            eval_metric=eval_metric,
            complexity=complexity,
            use_spark = context.use_spark
        )
        resul_dict = factory.run_all_model()

        best_model = ModelSelection(
            model_result = resul_dict,
            eval_metric = eval_metric
        ).select_best_model()

    except Exception as e:
        raise ModellingError(f"Modelling failed: {e}")
    
    log("Modelling", "end", "model training and evaluation completed")
    # ----------------------------
    # Statistics: Modelling Quality and Validation
    # ----------------------------
    evaluations = [
        RepeatedSubmodeling(),
        ParameterSensitivity(),
        FeaturesImportance(),
        PredictionSummary()
    ]
    model_evaluation_result = []

    for model_name, model_result in resul_dict.items():
        for eval_class in evaluations:
            model_eval_result = eval_class.run(model_result, df = df_preprocessed)
            model_evaluation_result.append(model_eval_result)
    # ----------------------------
    # Reporting
    # ----------------------------

    log("Reporting", "start", "start generating report into HTML")

    try:
        generate_html_report(
            api_url=context.api_url,
            df=df,
            normed_df=df_preprocessed,
            result_dict=resul_dict,
            logs= log_stream.getvalue(),
            use_spark=context.use_spark
        )
    except Exception as e:
        ReportingError(f"Report generation failed: {e}")
    
    log("Reporting", "end", "Reporting complered")
