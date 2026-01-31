import logging
from io import StringIO
from typing import Dict, List, Optional, Any, Union, Tuple, Literal

log_stream = StringIO()
base_logger = logging.getLogger("anomaly_logger")
base_logger.setLevel(logging.INFO)

if base_logger.hasHandlers():
    base_logger.handlers.clear()

stream_handler = logging.StreamHandler(log_stream)
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(step)s - %(stage)s - %(message)s"
)
stream_handler.setFormatter(formatter)
base_logger.addHandler(stream_handler)


def log(step: str, stage: str, message: str) -> None:
    extra: Dict[str, str] = {"step": step, "stage": stage}
    base_logger.info(message, extra=extra)
