import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.features import TREE_FEATURES, create_tree_features
from src.predict import BikeDemandPredictor

# Page configuration ---------------------------------------------------------------

st.set_page_config(
    page_title="Bike Demand Forecasting",
    page_icon="🚲",
    layout="wide",
)


PROJECT_ROOT = Path(__file__).resolve().parent


# Load predictor ---------------------------------------------------------------

@st.cache_resource
def load_predictor():

    return BikeDemandPredictor()


# Load historical data ---------------------------------------------------------------

@st.cache_data
def load_data():

    path = PROJECT_ROOT / "data" / "hour_cleaned.csv"

    df = pd.read_csv(
        path,
        parse_dates=["datetime"],
    )

    df = df.sort_values("datetime")

    df = df.set_index("datetime")

    return df


# Load metadata ---------------------------------------------------------------

@st.cache_data
def load_metadata():

    path = PROJECT_ROOT / "models" / "metadata.json"

    if not path.exists():
        return None

    with open(path, "r") as file:
        return json.load(file)


# Application ---------------------------------------------------------------

st.title("🚲 Bike Demand Forecasting")

st.write(
    "Forecast bike rental demand using the model "
    "selected by the training pipeline."
)


# Load model ---------------------------------------------------------------

try:

    predictor = load_predictor()

except Exception as error:

    st.error("Unable to load the trained model.")

    st.code(str(error))

    st.info(
        "Run the training pipeline first with ./run.sh"
    )

    st.stop()


# Load data ---------------------------------------------------------------

try:

    historical_data = load_data()

except Exception as error:

    st.error("Unable to load historical data.")

    st.code(str(error))

    st.stop()


# Model information ---------------------------------------------------------------

model_info = load_metadata()

st.write("production model ---------------------------------------------------------------")

st.metric(
    "Selected Model",
    predictor.model_name,
)


if model_info:

    with st.expander("Model metadata"):

        st.json(model_info)


# Forecast inputs ---------------------------------------------------------------

st.write("forecast inputs ---------------------------------------------------------------")

latest_timestamp = historical_data.index.max()

col1, col2 = st.columns(2)

with col1:

    forecast_date = st.date_input(
        "Forecast date",
        value=latest_timestamp.date(),
    )

with col2:

    forecast_hour = st.number_input(
        "Forecast hour",
        min_value=0,
        max_value=23,
        value=12,
        step=1,
)


forecast_timestamp = pd.Timestamp(
    forecast_date
).replace(
    hour=forecast_hour
)


# Weather ---------------------------------------------------------------

st.write("weather ---------------------------------------------------------------")

col1, col2, col3 = st.columns(3)

with col1:

    temp = st.number_input(
        "Temperature",
        min_value=-20.0,
        max_value=50.0,
        value=20.0,
        step=0.1,
    )

with col2:

    hum = st.number_input(
        "Humidity",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0,
    )

with col3:

    windspeed = st.number_input(
        "Wind speed",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.1,
    )


weathersit = st.selectbox(
    "Weather situation",
    options=[1, 2, 3, 4],
    format_func=lambda value: {
        1: "Clear",
        2: "Mist / Cloudy",
        3: "Light rain / Snow",
        4: "Heavy rain / Storm",
    }[value],
)


# Calendar ---------------------------------------------------------------

st.write("calendar ---------------------------------------------------------------")

col1, col2 = st.columns(2)

with col1:

    holiday = st.selectbox(
        "Holiday",
        options=[0, 1],
        format_func=lambda value:
            "Yes" if value == 1 else "No",
    )

with col2:

    workingday = st.selectbox(
        "Working day",
        options=[0, 1],
        format_func=lambda value:
            "Yes" if value == 1 else "No",
    )


# Prediction ---------------------------------------------------------------

st.write("prediction ---------------------------------------------------------------")

if st.button(
    "Predict Bike Demand",
    type="primary",
):

    # Copy historical data so the cached
    # DataFrame is never modified.
    prediction_data = historical_data.copy()

    # Create the future observation.
    future_row = pd.DataFrame(
        {
            "temp": [temp],
            "hum": [hum],
            "windspeed": [windspeed],
            "weathersit": [weathersit],
            "holiday": [holiday],
            "workingday": [workingday],
            "yr": [
                forecast_timestamp.year - 2011
            ],
            "cnt": [np.nan],
        },
        index=[forecast_timestamp],
    )

    # Add future observation to historical context.
    prediction_data = pd.concat(
        [
            prediction_data,
            future_row,
        ]
    )

    # Make sure timestamps are unique and ordered.
    prediction_data = (
        prediction_data
        .loc[
            ~prediction_data.index.duplicated(
                keep="last"
            )
        ]
        .sort_index()
    )

    # Build the exact feature row using the same engineered features the
    # LightGBM model was trained on. This prevents NaN lag/rolling features
    # from collapsing the model to a constant output.
    engineered = create_tree_features(prediction_data)
    feature_row = engineered[TREE_FEATURES].iloc[-1:].copy()

    target_series = historical_data["cnt"] if "cnt" in historical_data.columns else historical_data["target"]

    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        lag_name = f"lag_{lag}"
        if pd.isna(feature_row[lag_name].iloc[0]):
            if len(target_series) >= lag:
                feature_row[lag_name] = float(target_series.iloc[-lag])
            elif len(target_series) > 0:
                feature_row[lag_name] = float(target_series.iloc[-1])
            else:
                feature_row[lag_name] = 0.0

    for window in [24, 168]:
        mean_name = f"rolling_mean_{window}"
        std_name = f"rolling_std_{window}"

        if pd.isna(feature_row[mean_name].iloc[0]):
            recent = target_series.tail(window)
            feature_row[mean_name] = float(recent.mean()) if len(recent) else 0.0

        if pd.isna(feature_row[std_name].iloc[0]):
            recent = target_series.tail(window)
            feature_row[std_name] = float(recent.std(ddof=0)) if len(recent) > 1 else 0.0

    feature_row = feature_row[TREE_FEATURES].copy().fillna(0.0)

    try:

        prediction = predictor.predict(
            feature_row
        )

        st.success(
            "Prediction completed."
        )

        st.metric(
            "Predicted Bike Demand",
            f"{prediction:,.0f}",
        )

        st.write(
            f"Forecast timestamp: "
            f"{forecast_timestamp}"
        )

    except Exception as error:

        st.error(
            "Prediction failed."
        )

        st.code(str(error))


# Historical data ---------------------------------------------------------------

st.write("historical data ---------------------------------------------------------------")

with st.expander("View recent historical data"):

    st.dataframe(
        historical_data.tail(20),
        use_container_width=True,
    )
