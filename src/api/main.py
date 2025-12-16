import mlflow.sklearn
import numpy as np
from fastapi import FastAPI
from src.api.pydantic_models import (
    PredictionRequest, PredictionResponse
)

app = FastAPI(title="Credit Risk API")

model = mlflow.sklearn.load_model(
    "models:/credit-risk-model/Production"
)

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    X = np.array([[req.total_amount,
                   req.avg_amount,
                   req.txn_count,
                   req.amount_std]])

    prob = model.predict_proba(X)[0, 1]
    return {"risk_probability": prob}
