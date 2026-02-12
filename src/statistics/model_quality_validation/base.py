from abc import ABC, abstractmethod
from typing import Dict, Any

class ModelEvaluation(ABC):
    """
    Base class for all model evaluation checks.
    """
    @abstractmethod
    def run(
        self,
        model_result: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        """
        execute the evaluation check on the model result.

        returns:
        algorithm,
        metric,
        status,
        score,
        suggestion
        """

        pass