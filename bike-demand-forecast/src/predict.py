import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from huggingface_hub import hf_hub_download

from src.config import LSTM_WINDOW
from src.features import (
    TREE_FEATURES,
    LSTM_FEATURES,
    create_tree_features,
    create_time_features,
)
from src.utils import inverse_transform_target

# Define your Hugging Face repository ID
REPO_ID = "ZubairAli25266/bike-demand-forecast"

class BikeDemandPredictor:

    def __init__(self):
        self.metadata = self._load_metadata()
        self.model_name = self.metadata["selected_model"]
        self.model = self._load_model()
        self.scaler = self._load_scaler()

    # Metadata ----------------------------------
    def _load_metadata(self):
        # Load locally because metadata.json is hosted on GitHub, not Hugging Face
        metadata_path = Path("models/metadata.json")
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                "Ensure the 'models' folder is pushed to your GitHub repo."
            )
            
        with open(metadata_path, "r") as f:
            return json.load(f)
        
    # Model ------------------------------
    def _load_model(self):
        if self.model_name == "LightGBM":
            model_path = hf_hub_download(
                repo_id=REPO_ID, 
                filename="lightgbm.joblib"
            )
            return joblib.load(model_path)

        if self.model_name == "Random Forest":
            model_path = hf_hub_download(
                repo_id=REPO_ID, 
                filename="random_forest.joblib"
            )
            return joblib.load(model_path)

        if self.model_name == "LSTM":
            model_path = hf_hub_download(
                repo_id=REPO_ID, 
                filename="lstm.keras"
            )
            return tf.keras.models.load_model(model_path)

        raise ValueError(f"Unknown model: {self.model_name}")

    # Scaler ---------------------------
    def _load_scaler(self):
        if self.model_name == "LSTM":
            try:
                scaler_path = hf_hub_download(
                    repo_id=REPO_ID, 
                    filename="scaler.joblib"
                )
                return joblib.load(scaler_path)
            except Exception as e:
                raise FileNotFoundError(f"LSTM scaler not found on Hugging Face: {e}")
        return None

    # Tree prediction -----------------------------
    def predict_tree(self, historical_data):
        df = create_tree_features(historical_data)
        latest = df[TREE_FEATURES].iloc[-1:]

        if latest.isna().any().any():
            raise ValueError(
                "Not enough historical data to create lag/rolling features."
            )

        prediction = self.model.predict(latest)
        return float(prediction[0])
    
    # LSTM prediction ---------------------------
    def predict_lstm(self, historical_data):
        df = create_time_features(historical_data)
        df = df[LSTM_FEATURES].copy()

        if len(df) < LSTM_WINDOW:
            raise ValueError(
                f"LSTM requires at least {LSTM_WINDOW} historical hours."
            )

        scaled = self.scaler.transform(df)
        sequence = scaled[-LSTM_WINDOW:]
        X = np.expand_dims(sequence, axis=0)

        prediction_scaled = self.model.predict(X, verbose=0).flatten()[0]
        
        prediction = inverse_transform_target(
            np.array([prediction_scaled]),
            self.scaler
        )[0]

        return float(prediction)

    # Public prediction interface ............................
    def predict(self, historical_data):
        if not isinstance(historical_data, pd.DataFrame):
            raise TypeError("historical_data must be a pandas DataFrame.")

        if not isinstance(historical_data.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")

        if self.model_name == "LSTM":
            return self.predict_lstm(historical_data)

        return self.predict_tree(historical_data)