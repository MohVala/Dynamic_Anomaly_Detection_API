from typing import Dict, List, Any, Union

def flatten_json(y: Union[Dict[str, Any], List[Any]])-> Dict[str, Any]:
    out = {}

    def flatten(x: str, name: str ='')->None:
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f'{name}{a}_')
        elif isinstance(x,list):
            for i, a in enumerate(x):
                flatten(a,f'{name}{i}_')
        else:
            out[name[:-1]] = x
        
    flatten(y)
    return out