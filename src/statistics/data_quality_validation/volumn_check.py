from .base import QualityCheck

class VolumeCheck(QualityCheck):
    def run(self, df):
        excepted = self.config["data_quality"]["records_volume_count"]
        actual = df.count() if hasattr(df, "count") else len(df)

        status = "PASS" if actual == excepted or excepted == None else "FAIL"

        return {
            "name": "volum_check",
            "status": status,
            "metric": actual,
            "threshold": excepted,
            "message": f"Record count {actual}, expected {excepted}"
        }