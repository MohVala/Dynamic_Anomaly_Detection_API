from typing import Dict, Any, Union
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from .base import QualityCheck
from scipy.stats import skew

class DistributionSanityCheck(QualityCheck):
    """
    Perform statistical sanity validation.

    FAIL  -> zero variance OR dataset too small
    WARN  -> extreme skewness
    PASS  -> statistically stable
    """

    def run(self, df: Union[pd.DataFrame, DataFrame])->Dict[str, Any]:
        if not isinstance(df, pd.DataFrame):
            df = df.toPandas()

        # Data set size sanity:
        if len(df)<30:
            return{
                "satus":"FAIL",
                "details": f"Dataset too small (f{len(df)} rows) for reliable modelling."
            }
        
        # zero variance detections:
        zero_variance_cols = df.columns(df.var()==0)

        if len(zero_variance_cols)>0:
            return{
                "satus":"FAIL",
                "details": f"zero variance detected in columns {list(zero_variance_cols)}"
            }
        
        # skewness detection:
        skewness_values = df.apply(skew, nan_policy = "omit")
        extreme_skew_cols = skewness_values(abs(skewness_values)>5)

        if len(extreme_skew_cols)>0:
            return {
                "status": "WARN",
                "details": f"extreme skewness detected in {list(extreme_skew_cols)}"
            }
        
        return {
            "status": "PASS",
            "details": "Features distributions are statistically stable."
        }