import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    MODEL_DIR,
    METADATA_PATH,
    LSTM_PATH,
    SCALER_PATH,
    LSTM_WINDOW,
)

from src.features import (
    TREE_FEATURES,
    LSTM_FEATURES,
    create_tree_features,
    create_time_features,
)

from src.utils import (
    inverse_transform_target,
)


class BikeDemandPredictor:

    def __init__(self):

        self.metadata = self._load_metadata()

        self.model_name = (
            self.metadata["selected_model"]
        )

        self.model = self._load_model()

        self.scaler = self._load_scaler()

 
    # Metadata----------------------------------

    def _load_metadata(self):

        if not METADATA_PATH.exists():

            raise FileNotFoundError(
                "Model metadata not found. "
                "Run the training pipeline first."
            )

        with open(
            METADATA_PATH,
            "r"
        ) as f:

            return json.load(f)

        
    # Model ------------------------------

    def _load_model(self):

        if self.model_name == "LightGBM":

            return joblib.load(
                MODEL_DIR / "lightgbm.joblib"
            )

        if self.model_name == "Random Forest":

            return joblib.load(
                MODEL_DIR / "random_forest.joblib"
            )

        if self.model_name == "LSTM":

            return tf.keras.models.load_model(
                LSTM_PATH
            )

        raise ValueError(
            f"Unknown model: {self.model_name}"
        )

    # Scaler ---------------------------

    def _load_scaler(self):

        if self.model_name == "LSTM":

            if not SCALER_PATH.exists():

                raise FileNotFoundError(
                    "LSTM scaler not found."
                )

            return joblib.load(
                SCALER_PATH
            )

        return None

    # Tree prediction -----------------------------

    def predict_tree(
        self,
        historical_data
    ):

        df = create_tree_features(
            historical_data
        )

        latest = (
            df[TREE_FEATURES]
            .iloc[-1:]
        )

        if latest.isna().any().any():

            raise ValueError(
                "Not enough historical data "
                "to create lag/rolling features."
            )

        prediction = self.model.predict(
            latest
        )

        return float(
            prediction[0]
        )
    
    # LSTM prediction ---------------------------

    def predict_lstm(
        self,
        historical_data
    ):

        df = create_time_features(
            historical_data
        )

        df = df[LSTM_FEATURES].copy()

        if len(df) < LSTM_WINDOW:

            raise ValueError(
                f"LSTM requires at least "
                f"{LSTM_WINDOW} historical hours."
            )

        scaled = self.scaler.transform(
            df
        )

        sequence = scaled[
            -LSTM_WINDOW:
        ]

        X = np.expand_dims(
            sequence,
            axis=0
        )

        prediction_scaled = (
            self.model
            .predict(
                X,
                verbose=0
            )
            .flatten()[0]
        )

        prediction = (
            inverse_transform_target(
                np.array([
                    prediction_scaled
                ]),
                self.scaler
            )[0]
        )

        return float(
            prediction
        )

    # Public prediction interface ............................

    def predict(
        self,
        historical_data
    ):

        if not isinstance(
            historical_data,
            pd.DataFrame
        ):

            raise TypeError(
                "historical_data must be "
                "a pandas DataFrame."
            )

        if not isinstance(
            historical_data.index,
            pd.DatetimeIndex
        ):

            raise TypeError(
                "DataFrame index must be "
                "a DatetimeIndex."
            )

        if self.model_name == "LSTM":

            return self.predict_lstm(
                historical_data
            )

        return self.predict_tree(
            historical_data
        )