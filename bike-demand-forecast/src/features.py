import numpy as np
import pandas as pd


from src.config import TARGET_COLUMN


def resolve_target_column(df):
    """Support both production data (cnt) and test fixtures (target)."""
    for candidate in (TARGET_COLUMN, "target", "cnt"):
        if candidate in df.columns:
            return candidate
    raise KeyError(f"Target column not found. Expected one of: {TARGET_COLUMN}, 'target', 'cnt'")


TREE_FEATURES = [
    "temp",
    "hum",
    "windspeed",
    "weathersit",
    "workingday",
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
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "rolling_std_168",
]


LSTM_FEATURES = [
    TARGET_COLUMN,
    "temp",
    "hum",
    "windspeed",
    "weathersit",
    "holiday",
    "workingday",
    "yr",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def create_time_features(df):
    """
    Create calendar and cyclical time features.
    """

    df = df.copy()

    df["hour"] = df.index.hour

    df["day_of_week"] = df.index.dayofweek

    df["month"] = df.index.month

    df["day_of_year"] = df.index.dayofyear

    df["week_of_year"] = (
        df.index.isocalendar()
        .week
        .astype(int)
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Daily cycle
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Weekly cycle
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


def create_lag_features(df):
    """
    Create historical demand lag features.

    All lags use only previous observations.
    """

    df = df.copy()
    target_col = resolve_target_column(df)

    lags = [
        1,
        2,
        3,
        6,
        12,
        24,
        48,
        72,
        168,
    ]

    for lag in lags:

        df[f"lag_{lag}"] = (
            df[target_col]
            .shift(lag)
        )

    return df


def create_rolling_features(df):
    """
    Create rolling statistics using historical demand only.
    """

    df = df.copy()
    target_col = resolve_target_column(df)

    # Shift first so the current target is never included.
    historical_target = (
        df[target_col]
        .shift(1)
    )

    df["rolling_mean_24"] = (
        historical_target
        .rolling(window=24)
        .mean()
    )

    df["rolling_std_24"] = (
        historical_target
        .rolling(window=24)
        .std()
    )

    df["rolling_mean_168"] = (
        historical_target
        .rolling(window=168)
        .mean()
    )

    df["rolling_std_168"] = (
        historical_target
        .rolling(window=168)
        .std()
    )

    return df


def create_tree_features(df):
    """
    Complete feature-engineering pipeline
    for LightGBM and Random Forest.
    """

    df = df.copy()

    df = create_time_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    return df


def prepare_tree_dataset(df):
    """
    Create the final tabular dataset used by
    LightGBM and Random Forest.
    """

    df = create_tree_features(df)
    target_col = resolve_target_column(df)

    # Remove rows where lag/rolling features
    # cannot yet be calculated.
    df = df.dropna(
        subset=TREE_FEATURES + [target_col]
    )

    X = df[TREE_FEATURES].copy()

    y = df[target_col].copy()

    return X, y, df


def prepare_lstm_dataset(df):
    """
    Prepare the 12 features used by the LSTM.
    """

    df = df.copy()

    df = create_time_features(df)

    df = df.dropna(
        subset=LSTM_FEATURES
    )

    return df[LSTM_FEATURES].copy()