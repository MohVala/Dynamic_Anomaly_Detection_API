from typing import Dict, Any, Union
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from .base import QualityCheck

class CorrelationCheck(QualityCheck):
    """
    Detects extreme multicollinearity between features.
    WARN  -> if strong correlations detected
    FAIL  -> if dataset is degenerate (almost all features highly correlated)
    PASS  -> otherwise
    """

    def run(self, df: Union[pd.DataFrame, DataFrame])->Dict[str, Any]:
        
        if not isinstance(df, pd.DataFrame):
            df = df.toPandas()

        if df.shape[1]<2:
            return {
                "status": "PASS",
                "details": "Not enough features to cmpute correlation."
            }
        
        corr_matrix = df.corr().abs()

        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        high_corr_count = (upper_triangle>0.95).sum().sum()

        if high_corr_count == 0:
            return {
                "status": "PASS",
                "details": "No extreme correlation detected"
            }

        if high_corr_count > df.shape[1]*2:
            return {
                "status": "FAIL",
                "details": f"Dataset appears degenerate with {int(high_corr_count)} extreme correlations."
            }
        
        return {
                "status": "WARN",
                "details": f"{int(high_corr_count)} highly correlated feature pairs detected."
        }