import json

import pandas as pd

from src.config import (
    MODEL_DIR,
    METADATA_PATH,
)


RESULTS_PATH = (
    MODEL_DIR / "evaluation_results.csv"
)


def load_results():

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: "
            f"{RESULTS_PATH}"
        )

    return pd.read_csv(
        RESULTS_PATH
    )


def select_best_model(results):

    if "validation_mae" not in results.columns:
        raise ValueError(
            "evaluation_results.csv must contain "
            "'validation_mae'."
        )

    results = results.dropna(
        subset=["validation_mae"]
    )

    if results.empty:
        raise ValueError(
            "No valid validation results found."
        )

    best_row = results.loc[
        results["validation_mae"].idxmin()
    ]

    return best_row


def save_metadata(best_model, results):

    metadata = {
        "selected_model": best_model["model"],
        "selection_metric": "validation_mae",
        "validation_mae": float(
            best_model["validation_mae"]
        ),
        "validation_rmse": float(
            best_model["validation_rmse"]
        ),
        "test_mae": float(
            best_model["test_mae"]
        ),
        "test_rmse": float(
            best_model["test_rmse"]
        ),
        "candidates": (
            results[
                [
                    "model",
                    "validation_mae",
                    "validation_rmse",
                    "test_mae",
                    "test_rmse",
                ]
            ]
            .to_dict(orient="records")
        ),
    }

    with open(
        METADATA_PATH,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


def main():

    print(
        "Loading evaluation results..."
    )

    results = load_results()

    best_model = select_best_model(
        results
    )

    save_metadata(
        best_model,
        results
    )

    print(
        "\nProduction model:"
    )

    print(
        best_model["model"]
    )

    print(
        f"Validation MAE: "
        f"{best_model['validation_mae']:.2f}"
    )

    print(
        f"Validation RMSE: "
        f"{best_model['validation_rmse']:.2f}"
    )

    print(
        "\nSaved metadata:"
    )

    print(
        METADATA_PATH
    )


if __name__ == "__main__":
    main()