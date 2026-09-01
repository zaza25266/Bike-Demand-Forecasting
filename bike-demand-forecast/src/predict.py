# src/predict.py

import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from huggingface_hub import hf_hub_download

from src.config import (
    LSTM_WINDOW,
    METADATA_PATH,
    MODEL_DIR,
    SCALER_PATH,
)
from src.features import (
    TREE_FEATURES,
    LSTM_FEATURES,
    create_tree_features,
    create_time_features,
)
from src.utils import inverse_transform_target

REPO_ID = "ZubairAli25266/bike-demand-forecast"


class BikeDemandPredictor:
    def __init__(self):
        self.metadata = self._load_metadata()
        self.model_name = self.metadata["selected_model"]
        self.model = self._load_model()
        self.scaler = self._load_scaler()

    def _local_model_path(self, filename):
        candidate = MODEL_DIR / filename
        if candidate.exists():
            return candidate
        return None

    def _load_metadata(self):
        base_dir = Path(__file__).resolve().parent.parent
        metadata_path = base_dir / "models" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    def _load_model(self):
        if self.model_name == "LightGBM":
            local_model = self._local_model_path("lightgbm.joblib")
            if local_model is not None:
                return joblib.load(local_model)
            model_path = hf_hub_download(repo_id=REPO_ID, filename="lightgbm.joblib")
            return joblib.load(model_path)

        if self.model_name == "Random Forest":
            local_model = self._local_model_path("random_forest.joblib")
            if local_model is not None:
                return joblib.load(local_model)
            model_path = hf_hub_download(repo_id=REPO_ID, filename="random_forest.joblib")
            return joblib.load(model_path)

        if self.model_name == "LSTM":
            local_model = self._local_model_path("lstm.keras")
            if local_model is not None:
                return tf.keras.models.load_model(local_model)
            model_path = hf_hub_download(repo_id=REPO_ID, filename="lstm.keras")
            return tf.keras.models.load_model(model_path)

        raise ValueError(f"Unknown model: {self.model_name}")

    def _load_scaler(self):
        if self.model_name == "LSTM":
            if SCALER_PATH.exists():
                return joblib.load(SCALER_PATH)
            try:
                scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler.joblib")
                return joblib.load(scaler_path)
            except Exception as e:
                raise FileNotFoundError(f"LSTM scaler not found locally or on Hugging Face: {e}")
        return None

    def predict_tree(self, historical_data):
        df = historical_data.copy()

        if set(TREE_FEATURES).issubset(df.columns) and len(df.columns) == len(TREE_FEATURES):
            feature_df = df[TREE_FEATURES].copy()
            if feature_df.isna().any().any():
                feature_df = feature_df.fillna(0.0)
            prediction = self.model.predict(feature_df)
            return float(prediction[0])

        df = create_tree_features(df)
        latest = df[TREE_FEATURES].iloc[-1:].copy()
        if latest.isna().any().any():
            latest = latest.fillna(0.0)
        prediction = self.model.predict(latest)
        return float(prediction[0])

    def predict_lstm(self, historical_data):
        df = create_time_features(historical_data)
        df = df[LSTM_FEATURES].copy()
        if len(df) < LSTM_WINDOW:
            raise ValueError(f"LSTM requires at least {LSTM_WINDOW} historical hours.")

        scaled = self.scaler.transform(df)
        sequence = scaled[-LSTM_WINDOW:]
        X = np.expand_dims(sequence, axis=0)
        prediction_scaled = self.model.predict(X, verbose=0).flatten()[0]
        prediction = inverse_transform_target(np.array([prediction_scaled]), self.scaler)[0]
        return float(prediction)

    def predict(self, historical_data):
        if not isinstance(historical_data, pd.DataFrame):
            raise TypeError("historical_data must be a pandas DataFrame.")
        if not isinstance(historical_data.index, pd.DatetimeIndex) and len(historical_data.columns) != len(TREE_FEATURES):
            raise TypeError("DataFrame index must be a DatetimeIndex.")

        if self.model_name == "LSTM":
            return self.predict_lstm(historical_data)
        return self.predict_tree(historical_data)
