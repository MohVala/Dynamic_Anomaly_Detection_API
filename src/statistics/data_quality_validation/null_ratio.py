from .base import QualityCheck

class NullRatioCheck(QualityCheck):
    def run(self, df):
        df_len = len(df)
        ratio_allowd = 0.1 # self.config["data_quality"]["max_null_ratio"]
        col_cnt = len(df.columns)
        null_cnt_avg = 0
        for col in df.columns:
            null_cnt = df[col].isnull().sum()
            null_cnt_avg += null_cnt
        ratio = null_cnt_avg / df_len
        
        status = "PASS" if ratio < ratio_allowd else "FAIL"

        return {
            "name": "null_ration_check",
            "status": status,
            "metric": ratio,
            "threshold": ratio_allowd,
            "message": f"Null ratio accross all features is {ratio:.2%}, allowed {ratio_allowd:.2%}."
        }