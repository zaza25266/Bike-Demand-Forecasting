import joblib
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

from src.config import (
    PROCESSED_DATA_PATH,
    MODEL_DIR,
    FEATURES_PATH,
    SCALER_PATH,
    LSTM_PATH,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    LSTM_WINDOW,
    LSTM_FEATURE_COUNT,
    RANDOM_STATE,
)

from src.features import (
    TREE_FEATURES,
    LSTM_FEATURES,
    prepare_tree_dataset,
    prepare_lstm_dataset,
)

from src.models import (
    build_lightgbm,
    build_random_forest,
    build_lstm,
)

from src.utils import (
    chronological_split,
    create_sequences,
)


MLFLOW_EXPERIMENT = "bike-demand-forecast"


def load_data():

    df = pd.read_csv(
        PROCESSED_DATA_PATH,
        parse_dates=["datetime"]
    )

    return (
        df.set_index("datetime")
        .sort_index()
    )


def train_lightgbm(
    X_train,
    y_train,
):

    model = build_lightgbm()

    with mlflow.start_run(
        run_name="LightGBM_training"
    ):

        model.fit(
            X_train,
            y_train
        )

        mlflow.log_param(
            "model",
            "LightGBM"
        )

        mlflow.log_param(
            "features",
            len(TREE_FEATURES)
        )

        mlflow.log_param(
            "train_rows",
            len(X_train)
        )

        mlflow.sklearn.log_model(
            model,
            "model",
            skops_trusted_types=[
                "collections.OrderedDict",
                "lightgbm.basic.Booster",
                "lightgbm.sklearn.LGBMRegressor",
            ],
        )

    joblib.dump(
        model,
        MODEL_DIR / "lightgbm.joblib"
    )

    return model


def train_random_forest(
    X_train,
    y_train,
):

    model = build_random_forest()

    with mlflow.start_run(
        run_name="RandomForest_training"
    ):

        model.fit(
            X_train,
            y_train
        )

        mlflow.log_param(
            "model",
            "Random Forest"
        )

        mlflow.log_param(
            "features",
            len(TREE_FEATURES)
        )

        mlflow.log_param(
            "train_rows",
            len(X_train)
        )

        mlflow.sklearn.log_model(
            model,
            "model",
            skops_trusted_types=[
                "collections.OrderedDict",
                "sklearn.ensemble._forest.RandomForestRegressor",
                "numpy.ndarray",
            ],
        )

    joblib.dump(
        model,
        MODEL_DIR / "random_forest.joblib"
    )

    return model


def train_lstm(df):

    data = prepare_lstm_dataset(df)

    n = len(data)

    train_end = int(
        n * TRAIN_RATIO
    )

    validation_end = int(
        n * (
            TRAIN_RATIO
            + VALIDATION_RATIO
        )
    )

    scaler = StandardScaler()

    # Fit scaler ONLY on training data.
    scaler.fit(
        data.iloc[:train_end]
    )

    scaled = scaler.transform(data)


    # Training sequences ----------------------------------------------


    train_data = scaled[
        :train_end
    ]

    X_train, y_train = create_sequences(
        train_data,
        LSTM_WINDOW
    )


    # Validation sequences -------------------------------------------


    validation_data = scaled[
        train_end - LSTM_WINDOW:
        validation_end
    ]

    X_val, y_val = create_sequences(
        validation_data,
        LSTM_WINDOW
    )


    # Build model --------------------------------------------------

    model = build_lstm(
        window=LSTM_WINDOW,
        n_features=LSTM_FEATURE_COUNT
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    with mlflow.start_run(
        run_name="LSTM_training"
    ):

        history = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_val,
                y_val
            ),
            epochs=30,
            batch_size=64,
            callbacks=[
                early_stopping
            ],
            verbose=1
        )

        mlflow.log_param(
            "model",
            "LSTM"
        )

        mlflow.log_param(
            "window",
            LSTM_WINDOW
        )

        mlflow.log_param(
            "features",
            LSTM_FEATURE_COUNT
        )

        mlflow.log_param(
            "batch_size",
            64
        )

        mlflow.log_metric(
            "best_val_loss",
            float(
                min(
                    history.history["val_loss"]
                )
            )
        )

        mlflow.tensorflow.log_model(
            model,
            "model"
        )

    model.save(
        LSTM_PATH
    )

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    return model


def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    print(
        "Loading processed data..."
    )

    df = load_data()

    # Tree models --------------------------------------------------------

    print(
        "\nPreparing tree-model features..."
    )

    X, y, _ = prepare_tree_dataset(df)

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
        TRAIN_RATIO,
        VALIDATION_RATIO
    )

    joblib.dump(
        TREE_FEATURES,
        FEATURES_PATH
    )

    print(
        "Training LightGBM..."
    )

    train_lightgbm(
        X_train,
        y_train
    )

    print(
        "Training Random Forest..."
    )

    train_random_forest(
        X_train,
        y_train
    )

  
    # LSTM ----------------------------------------------------
  

    print(
        "\nTraining LSTM..."
    )

    train_lstm(df)

    print(
        "\nAll model training complete."
    )


if __name__ == "__main__":
    main()