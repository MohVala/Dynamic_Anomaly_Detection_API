import pandas as pd
from typing import Dict, Any, Union
from pyspark.sql import DataFrame
from .base import QualityCheck

class DupicatedCheck(QualityCheck):
    """ this is class to check is there any remain duplicate or not."""

    def run(self, df: Union[pd.DataFrame, DataFrame])->Dict[str, Any]:
        duplicate_cnt = df.duplicate().sum()

        if duplicate_cnt>0:
            status = "WARN"
            return {
                "status": status,
                "details": f"There are {duplicate_cnt} duplicated records. "
            }
        status = "PASS"
        return {
                "status": status,
                "details": "There are no duplicated records. "
            }
