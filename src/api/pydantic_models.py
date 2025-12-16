from pydantic import BaseModel


class PredictionRequest(BaseModel):
    total_amount: float
    avg_amount: float
    txn_count: int
    amount_std: float


class PredictionResponse(BaseModel):
    risk_probability: float
