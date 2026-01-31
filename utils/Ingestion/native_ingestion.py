from typing import Dict, List, Any, Union
import pandas as pd
import requests
from ..logger import log
from .api_flatten import flatten_json


def ingest_api_to_dataframe(api_url: str) -> pd.DataFrame:
    try:
        log("data_ingestion", "start", "Ingesting API data into Pandas DataFrame")
        response = requests.get(api_url)
        response.raise_for_status()
        raw_json: Union[Dict[str, Any], List[Any]] = response.json()

        # handle list or single object:
        if isinstance(raw_json, list):
            data: List[Dict[str, Any]] = [flatten_json(item) for item in raw_json]
        else:
            data = [flatten_json(raw_json)]

        df = pd.DataFrame(data)
        log("data_ingestion", "end", f"Pandas DataFrame created with {len(df)} rows")
        return df

    except Exception as e:
        log("data_ingestion", "error", str(e))
        return pd.DataFrame()
