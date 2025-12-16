import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


##Custom Transformers

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["TransactionHour"] = X["TransactionStartTime"].dt.hour
        X["TransactionDay"] = X["TransactionStartTime"].dt.day
        X["TransactionMonth"] = X["TransactionStartTime"].dt.month
        X["TransactionYear"] = X["TransactionStartTime"].dt.year
        return X


class CustomerAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby("CustomerId").agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            txn_count=("TransactionId", "count"),
            amount_std=("Amount", "std"),
        ).reset_index()

        agg["amount_std"] = agg["amount_std"].fillna(0)
        return agg


##Build Feature Pipeline

def build_feature_pipeline():
    numeric_features = [
        "total_amount",
        "avg_amount",
        "txn_count",
        "amount_std"
    ]

    categorical_features = []

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor

##End-to-End Processing Function

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df = DateFeatureExtractor().transform(df)
    agg = CustomerAggregator().transform(df)

    return agg
