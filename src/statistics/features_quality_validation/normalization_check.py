import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from pyspark.sql import DataFrame
from .base import QualityCheck

class NormalizationCheck(QualityCheck):
    """ to check features are normalize properly scaled between 0 to 1"""

    def run(
            self, df: Union[pd.DataFrame, DataFrame]
            )->Dict[str, Any]:
        numeric_col = df.select_dtypes(included = [np.number]).columns
        
        for col in numeric_col:
            min_val = df[col].min()
            max_val = df[col].max()

            if min_val<0 or max_val>1:
                status = "FAIL"
                return {
                    "status": status,
                    "details": "data is not scaled properly."
                }
            status = "PASS"
            return {
                    "status": status,
                    "details": "data is normalized correctly."
            }