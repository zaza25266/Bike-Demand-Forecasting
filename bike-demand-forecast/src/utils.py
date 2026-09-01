import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def chronological_split(
    X,
    y,
    train_ratio=0.70,
    validation_ratio=0.15,
):
    """
    Chronological train/validation/test split.

    No shuffling is performed because this is time-series data.
    """

    n = len(X)

    train_end = int(
        n * train_ratio
    )

    validation_end = int(
        n * (train_ratio + validation_ratio)
    )

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:validation_end]
    X_test = X.iloc[validation_end:]

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:validation_end]
    y_test = y.iloc[validation_end:]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def create_sequences(
    data,
    window,
    target_index=0,
):
    """
    Convert time-series data into sliding windows.

    Example:

    window = 168

    X[i] = previous 168 hours
    y[i] = target immediately after that window
    """

    X = []
    y = []

    for i in range(
        window,
        len(data)
    ):

        X.append(
            data[i - window:i]
        )

        y.append(
            data[i, target_index]
        )

    return (
        np.asarray(X),
        np.asarray(y).reshape(-1, 1)
    )


def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate MAE and RMSE.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
    }


def inverse_transform_target(
    values,
    scaler,
    target_index=0,
):
    """
    Convert scaled target values back
    to the original demand units.
    """

    return (
        values * scaler.scale_[target_index]
        + scaler.mean_[target_index]
    )