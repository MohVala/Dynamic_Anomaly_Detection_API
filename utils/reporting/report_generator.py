from typing import Union
from .native_report_generator import native_generate_html_report
from .spark_report_generator import spark_generate_html_report
def generate_html_report(
    api_url,
    df,
    normed_df,
    result_dict,
    logs,
    use_spark: bool
):
    if use_spark:
        spark_generate_html_report(api_url=api_url,
            df=df,
            normed_df=normed_df,
            result_dict=result_dict,
            logs=logs,
            )
    else:
        native_generate_html_report(api_url=api_url,
            df=df,
            normed_df=normed_df,
            result_dict=result_dict,
            logs=logs,
            )