from .base import QualityCheck

class ColumnCountCheck(QualityCheck):
    def run(self, df):
        excepted = self.config["data_quality"]["expected_column_count"]
        actual = len(df.columns)

        status = "PASS" if actual == excepted or excepted == None else "FAIL"

        return {
            "name" : "coumns_count_check",
            "status": status,
            "metric": actual,
            "threshold": excepted,
            "message": f"Expected {excepted} columns, got {actual}"
        }