import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import sum, when, col
from typing import Dict, Any, Union
from .base import QualityCheck

class MissingCheck(QualityCheck):
    """This is to check is there any missing values or not."""

    def run(self,df:Union[pd.DataFrame, DataFrame])->Dict[str, Any]:
        if isinstance(df, pd.DataFrame):

            missing_cnt = df.isnull().sum().sum()
        else:
            missing_cnt = df.select(
                [
                    sum(when(col(c).isNull(),1).otherwise(0).alias(c))
                    for c in df.columns
                ]
            ).collect()[0].asDict()
            missing_cnt = sum(missing_cnt.values())
            
        if missing_cnt>0:
            status = "FAIL"

            return {
                "status":status,
                "details": f"There are {missing_cnt} missing value which not handled."
            }
        
        status = "PASS"
        return {
                "status":status,
                "details": "all missing values handle by KMEANS."
            }