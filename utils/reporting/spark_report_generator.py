import pandas as pd
from .native_report_generator import native_generate_html_report


def spark_generate_html_report(
    api_url: str,
    df,
    normed_df,
    result_dict: dict,
    logs: str,
):
    """
    Spark reporting strategy:
    - Convert Spark DataFrames to Pandas
    - Reuse native reporting logic
    """

    df_pd = df.toPandas()
    normed_df_pd = normed_df.toPandas()

    # Normalize anomaly outputs if needed
    for model in result_dict.values():
        anomalies = model["anomaly_detection"]
        if hasattr(anomalies, "toPandas"):
            model["anomaly_detection"] = anomalies.toPandas().iloc[:, 0].values

    native_generate_html_report(
        api_url=api_url,
        df=df_pd,
        normed_df=normed_df_pd,
        result_dict=result_dict,
        logs=logs,
    )
