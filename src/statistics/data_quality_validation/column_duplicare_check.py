from .base import QualityCheck

class ColumnDuplicateCheck(QualityCheck):
    def run(self, df):
        duplicate_column_cnt = 0
        duplicate_column_allow = 0 # self.config["data_quality"]["columns_duplicate_allow"]
        columns = df.columns
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                if df[i].equals(df[j]):
                    duplicate_column_cnt += 1
        
        status = "PASS" if duplicate_column_cnt <= duplicate_column_allow else "WARN"

        return {
            "name": "column_duplicate_check",
            "status": status,
            "metric": duplicate_column_cnt,
            "threshold": duplicate_column_allow,
            "message": f"{duplicate_column_cnt} duplicate detected, allowd {duplicate_column_allow}."
        }
            