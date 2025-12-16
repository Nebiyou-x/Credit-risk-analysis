import pandas as pd
from src.data_processing import process_data, create_rfm_target


def test_process_data_columns():
    df = pd.DataFrame(
        {
            "TransactionId": ["t1", "t2"],
            "CustomerId": ["c1", "c1"],
            "Amount": [100, 200],
            "Value": [100, 200],
            "TransactionStartTime": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        }
    )

    result = process_data(df)
    expected_cols = {
        "CustomerId",
        "total_amount",
        "avg_amount",
        "txn_count",
        "amount_std",
    }

    assert expected_cols.issubset(result.columns)


def test_create_rfm_target():
    df = pd.DataFrame(
        {
            "TransactionId": ["t1", "t2", "t3"],
            "CustomerId": ["c1", "c2", "c3"],
            "Value": [100, 10, 50],
            "TransactionStartTime": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03"]
            ),
        }
    )

    target = create_rfm_target(df)

    assert "is_high_risk" in target.columns
    assert target["is_high_risk"].isin([0, 1]).all()
    assert len(target) == 3
