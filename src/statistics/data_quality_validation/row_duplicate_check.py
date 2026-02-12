from .base import QualityCheck

class RowDuplicateCheck(QualityCheck):
    def run(self, df):
        df_len = len(df)
        duplicate_allow = 0.1 # self.confi["data_quality"]["max_duplicated_records_ratio"]
        duplicates = df.duplicated().sum()

        ratio = duplicates / df_len

        status = "PASS" if ratio >= duplicate_allow else "FAIL"

        return {
            "name": "row_duplicate_check",
            "status": status,
            "metric": ratio,
            "threshold": duplicate_allow,
            "message": f"duplicated record ratio is {ratio:.2%}, allowd is {duplicate_allow:.2%}."
        }