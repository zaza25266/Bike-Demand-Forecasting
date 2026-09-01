

import numpy as np
import pandas as pd

from src.features import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    create_tree_features,
    TREE_FEATURES,
)

from src.utils import (
    chronological_split,
    create_sequences,
    calculate_metrics,
)

from src.model_selection import (
    select_best_model,
)


def make_sample_data(rows=300):

    index = pd.date_range(
        start="2024-01-01",
        periods=rows,
        freq="h",
    )

    df = pd.DataFrame(
        {
            "target": np.arange(rows).astype(float),
            "temp": np.random.rand(rows),
            "hum": np.random.rand(rows),
            "windspeed": np.random.rand(rows),
            "weathersit": np.ones(rows),
            "holiday": np.zeros(rows),
            "workingday": np.ones(rows),
            "yr": np.ones(rows),
        },
        index=index,
    )

    return df


def test_time_features():

    df = make_sample_data()

    result = create_time_features(df)

    expected_columns = [
        "hour",
        "day_of_week",
        "month",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    for column in expected_columns:

        assert column in result.columns


def test_lag_features():

    df = make_sample_data()

    result = create_lag_features(df)

    assert "lag_1" in result.columns
    assert "lag_24" in result.columns
    assert "lag_168" in result.columns

    # lag_1 at time t must equal
    # target at time t-1
    assert (
        result["lag_1"].iloc[1]
        == result["target"].iloc[0]
    )


def test_rolling_features_do_not_use_current_target():

    df = make_sample_data()

    result = create_rolling_features(df)

    # At the first row there is no
    # previous observation.
    assert pd.isna(
        result["rolling_mean_24"].iloc[0]
    )

    # The rolling mean at t=24 should
    # use observations before t=24.
    expected = (
        df["target"]
        .iloc[0:24]
        .mean()
    )

    actual = (
        result["rolling_mean_24"]
        .iloc[24]
    )

    assert actual == expected


def test_tree_features():

    df = make_sample_data()

    result = create_tree_features(df)

    for feature in TREE_FEATURES:

        assert feature in result.columns


def test_chronological_split():

    X = pd.DataFrame(
        {
            "feature": np.arange(100)
        }
    )

    y = pd.Series(
        np.arange(100)
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = chronological_split(
        X,
        y,
        train_ratio=0.70,
        validation_ratio=0.15,
    )

    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15

    # Ensure chronological ordering.
    assert X_train.index.max() < X_val.index.min()
    assert X_val.index.max() < X_test.index.min()


def test_create_sequences():

    data = np.arange(20).reshape(
        10,
        2
    )

    X, y = create_sequences(
        data,
        window=3
    )

    assert X.shape == (
        7,
        3,
        2
    )

    assert y.shape == (
        7,
        1
    )

    # First sequence should contain
    # rows 0, 1, 2.
    np.testing.assert_array_equal(
        X[0],
        data[0:3]
    )

    # Target is row 3.
    assert y[0, 0] == data[3, 0]


def test_metrics():

    y_true = np.array(
        [10, 20, 30]
    )

    y_pred = np.array(
        [12, 18, 33]
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    assert "mae" in metrics
    assert "rmse" in metrics

    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0


def test_model_selection():

    results = pd.DataFrame(
        {
            "model": [
                "LightGBM",
                "Random Forest",
                "LSTM",
            ],
            "validation_mae": [
                35.0,
                42.0,
                37.0,
            ],
            "validation_rmse": [
                50.0,
                60.0,
                53.0,
            ],
            "test_mae": [
                34.0,
                41.0,
                36.0,
            ],
            "test_rmse": [
                49.0,
                59.0,
                52.0,
            ],
        }
    )

    best = select_best_model(
        results
    )

    assert best["model"] == "LightGBM"