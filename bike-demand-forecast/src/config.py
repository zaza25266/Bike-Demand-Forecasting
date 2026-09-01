
from pathlib import Path


# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent


# Directories
DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


# Data
RAW_DATA_PATH = DATA_DIR / "hour.csv"
PROCESSED_DATA_PATH = DATA_DIR / "hour_cleaned.csv"


# Models
LIGHTGBM_PATH = MODEL_DIR / "lightgbm.joblib"
RANDOM_FOREST_PATH = MODEL_DIR / "random_forest.joblib"
LSTM_PATH = MODEL_DIR / "lstm.keras"

FEATURES_PATH = MODEL_DIR / "features.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


# Target
TARGET_COLUMN = "cnt"


# LSTM configuration
LSTM_WINDOW = 168
LSTM_FEATURE_COUNT = 12


# Data split
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# Random state
RANDOM_STATE = 42

