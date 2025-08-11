from pydantic import BaseModel

class Response(BaseModel):
    probs: list = []
    best_prob: float = -1.0
    pred_id: int = -1
    pred_class: str = ""
    pred_name: str = ""

