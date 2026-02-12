class DecisionEngine:
    def __init__(self, fail_fast: bool = True):
        self.fail_fast = fail_fast

    def decide(self, results):
        statuses = [res["status"] for res in results]

        if "FAIL" in statuses:
            return {
                "final_status": "FAIL",
                "detials": results
            }
        
        if "WARN" in statuses:
            return {
                "final_status": "WARN",
                "details": results
            }
        
        return{
            "final_status": "PASS",
            "details": results
        }