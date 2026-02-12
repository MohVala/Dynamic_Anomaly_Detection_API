from typing import Dict, Any

class ModelSelection:
    @staticmethod
    def select_best_model(
        model_result: Dict[str, Dict[str, Any]],
        eval_metric:str
    )->Dict[str, Any]:
        
        best_model = None
        best_score = None

        for method, result in model_result.items():
            score = result["score"]
            if eval_metric == "davies_bouldin":
                if best_score is None or score < best_score:
                    best_score  = score
                    best_model = result
            else:
                if best_score is None or score>best_score:
                    best_score  = score
                    best_model = result
        
        return best_model