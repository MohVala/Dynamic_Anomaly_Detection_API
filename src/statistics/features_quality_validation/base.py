from abc import ABC, abstractmethod
from typing import Dict, Any

class QualityCheck(ABC):
    """
    Abstract base class for feature quality validation checks.
    All features checks should inherit from this class and should implement 'run'.
    """

    def __init__(self, config: Dict[str, Any]= None):
        self.config = config or {}

    @abstractmethod
    def run(self, df) -> Dict[str, Any]:
        """
        Execute the quality check on the dataframes (pandas and dataframe).
        return:
        Dict[str, Any]: Must include at leas 'status' (PASS/WARN/FAIL) and 'details'.
        """
        pass