import joblib
import mlflow
import pandas as pd
import tensorflow as tf

from src.config import (
    PROCESSED_DATA_PATH,
    MODEL_DIR,
    LSTM_PATH,
    SCALER_PATH,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    LSTM_WINDOW,
)

from src.features import (
    prepare_tree_dataset,
    prepare_lstm_dataset,
)

from src.utils import (
    chronological_split,
    create_sequences,
    calculate_metrics,
    inverse_transform_target,
)


def load_data():

    df = pd.read_csv(
        PROCESSED_DATA_PATH,
        parse_dates=["datetime"]
    )

    return (
        df.set_index("datetime")
        .sort_index()
    )


def evaluate_tree_model(
    model_name,
    model_path,
    X_val,
    X_test,
    y_val,
    y_test,
):

    model = joblib.load(
        model_path
    )

    validation_predictions = model.predict(
        X_val
    )

    test_predictions = model.predict(
        X_test
    )

    validation_metrics = calculate_metrics(
        y_val,
        validation_predictions
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    with mlflow.start_run(
        run_name=f"{model_name}_evaluation"
    ):

        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_metric(
            "validation_mae",
            validation_metrics["mae"]
        )

        mlflow.log_metric(
            "validation_rmse",
            validation_metrics["rmse"]
        )

        mlflow.log_metric(
            "test_mae",
            test_metrics["mae"]
        )

        mlflow.log_metric(
            "test_rmse",
            test_metrics["rmse"]
        )

    return {
        "model": model_name,
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
    }


def evaluate_lstm(df):

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

    scaler = joblib.load(
        SCALER_PATH
    )

    scaled = scaler.transform(data)


    # Validation ---------------------------------

    validation_data = scaled[
        train_end - LSTM_WINDOW:
        validation_end
    ]

    X_val, y_val = create_sequences(
        validation_data,
        LSTM_WINDOW
    )

    # Test --------------------------------

    test_data = scaled[
        validation_end - LSTM_WINDOW:
    ]

    X_test, y_test = create_sequences(
        test_data,
        LSTM_WINDOW
    )

    model = tf.keras.models.load_model(
        LSTM_PATH
    )

    validation_predictions = model.predict(
        X_val,
        verbose=0
    ).flatten()

    test_predictions = model.predict(
        X_test,
        verbose=0
    ).flatten()

    # Convert back to actual demand.
    y_val = inverse_transform_target(
        y_val.flatten(),
        scaler
    )

    y_test = inverse_transform_target(
        y_test.flatten(),
        scaler
    )

    validation_predictions = inverse_transform_target(
        validation_predictions,
        scaler
    )

    test_predictions = inverse_transform_target(
        test_predictions,
        scaler
    )

    validation_metrics = calculate_metrics(
        y_val,
        validation_predictions
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    with mlflow.start_run(
        run_name="LSTM_evaluation"
    ):

        mlflow.log_param(
            "model",
            "LSTM"
        )

        mlflow.log_metric(
            "validation_mae",
            validation_metrics["mae"]
        )

        mlflow.log_metric(
            "validation_rmse",
            validation_metrics["rmse"]
        )

        mlflow.log_metric(
            "test_mae",
            test_metrics["mae"]
        )

        mlflow.log_metric(
            "test_rmse",
            test_metrics["rmse"]
        )

    return {
        "model": "LSTM",
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
    }


def main():

    mlflow.set_experiment(
        "bike-demand-forecast"
    )

    df = load_data()

    # Tree models -----------------------------------

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

    results = []

    results.append(
        evaluate_tree_model(
            "LightGBM",
            MODEL_DIR / "lightgbm.joblib",
            X_val,
            X_test,
            y_val,
            y_test,
        )
    )

    results.append(
        evaluate_tree_model(
            "Random Forest",
            MODEL_DIR / "random_forest.joblib",
            X_val,
            X_test,
            y_val,
            y_test,
        )
    )

  
    # LSTM ------------------------------


    results.append(
        evaluate_lstm(df)
    )

   
    # Save results -----------------------------
 

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        MODEL_DIR / "evaluation_results.csv",
        index=False
    )

    print(
        "\nModel evaluation:"
    )

    print(
        results_df
        .sort_values("validation_mae")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()