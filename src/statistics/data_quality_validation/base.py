from abc import ABC, abstractmethod
from typing import Dict, Any

class QualityCheck(ABC):
    """
    Abstract base class for raw data quality validation.
    All raw data checks should inherit from this class and implement "run"
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def run(self, df) -> Dict[str, Any]:
        """
        Execute Quality and Validation check.
        """
        pass
