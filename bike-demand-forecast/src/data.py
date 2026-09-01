import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)


def load_raw_data():

    df = pd.read_csv(RAW_DATA_PATH)

    df["datetime"] = pd.to_datetime(
        df["dteday"]
    ) + pd.to_timedelta(
        df["hr"],
        unit="h"
    )

    df = (
        df.set_index("datetime")
          .sort_index()
    )

    return df





def restore_hourly_frequency(df):

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h"
    )

    df = df.reindex(full_index)

    df.index.name = "datetime"

    return df


def handle_missing_values(df):

    # Target
    df["cnt"] = (
        df["cnt"]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    # Continuous variables
    continuous = [
        "temp",
        "hum",
        "windspeed"
    ]

    df[continuous] = (
        df[continuous]
        .interpolate(method="time")
        .ffill()
        .bfill()
    )

    # Discrete/categorical variables
    discrete = [
        "weathersit",
        "holiday",
        "workingday",
        "yr"
    ]

    df[discrete] = (
        df[discrete]
        .ffill()
        .bfill()
    )

    return df


def save_processed_data(df):

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(PROCESSED_DATA_PATH)


def main():

    print("Loading raw data...")

    df = load_raw_data()

    print("Original shape:", df.shape)

    df = restore_hourly_frequency(df)

    print("After restoring hourly frequency:", df.shape)

    df = handle_missing_values(df)

    print(
        "Remaining missing values:",
        df.isna().sum().sum()
    )

    save_processed_data(df)

    print(
        "Saved:",
        PROCESSED_DATA_PATH
    )


if __name__ == "__main__":
    main()