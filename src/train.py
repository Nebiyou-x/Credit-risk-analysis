import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

from src.data_processing import process_data, create_rfm_target


def evaluate(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

def train():
    df = pd.read_csv("data/raw/data.csv")
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    features = process_data(df)
    target = create_rfm_target(df)

    data = features.merge(target, on="CustomerId")

    X = data.drop(columns=["CustomerId", "is_high_risk"])
    y = data["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        )
    }

    mlflow.set_experiment("credit-risk-model")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = evaluate(y_test, y_pred, y_prob)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
if __name__ == "__main__":
    train()
