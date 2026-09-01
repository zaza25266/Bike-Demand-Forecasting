# app.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.features import TREE_FEATURES, create_tree_features
from src.predict import BikeDemandPredictor

st.set_page_config(
    page_title="Bike Demand Forecasting",
    page_icon="🚲",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_predictor():
    return BikeDemandPredictor()


@st.cache_data
def load_data():
    path = PROJECT_ROOT / "data" / "hour_cleaned.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime")
    df = df.set_index("datetime")
    return df


@st.cache_data
def load_metadata():
    path = PROJECT_ROOT / "models" / "metadata.json"
    if not path.exists():
        return None
    with open(path, "r") as file:
        return json.load(file)


st.title("🚲 Bike Demand Forecasting")
st.write("Forecast bike rental demand using the model selected by the training pipeline.")

try:
    predictor = load_predictor()
except Exception as error:
    st.error("Unable to load the trained model.")
    st.code(str(error))
    st.info("Run the training pipeline first with ./run.sh")
    st.stop()

try:
    historical_data = load_data()
except Exception as error:
    st.error("Unable to load historical data.")
    st.code(str(error))
    st.stop()

model_info = load_metadata()
st.write("production model ---------------------------------------------------------------")
st.metric("Selected Model", predictor.model_name)

if model_info:
    with st.expander("Model metadata"):
        st.json(model_info)

st.write("forecast inputs ---------------------------------------------------------------")
latest_timestamp = historical_data.index.max()
default_timestamp = latest_timestamp + pd.Timedelta(hours=1)

col1, col2 = st.columns(2)
with col1:
    forecast_date = st.date_input("Forecast date", value=default_timestamp.date())
with col2:
    forecast_hour = st.number_input(
        "Forecast hour",
        min_value=0,
        max_value=23,
        value=int(default_timestamp.hour),
        step=1,
    )

forecast_timestamp = pd.Timestamp(forecast_date).replace(hour=forecast_hour)

if forecast_timestamp <= latest_timestamp:
    st.warning(
        f"Selected timestamp ({forecast_timestamp}) is at or before the last "
        f"historical timestamp ({latest_timestamp}). Choose a later timestamp."
    )
    st.stop()

st.write("weather ---------------------------------------------------------------")
col1, col2, col3 = st.columns(3)
with col1:
    temp = st.number_input("Temperature", min_value=-20.0, max_value=50.0, value=20.0, step=0.1)
with col2:
    hum = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
with col3:
    windspeed = st.number_input("Wind speed", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

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

st.write("calendar ---------------------------------------------------------------")
workingday = st.selectbox(
    "Working day",
    options=[0, 1],
    format_func=lambda value: "Yes" if value == 1 else "No",
)

st.write("prediction ---------------------------------------------------------------")
if st.button("Predict Bike Demand", type="primary"):

    prediction_data = historical_data.copy()
    target_col = "cnt" if "cnt" in prediction_data.columns else "target"

    hours_ahead = int((forecast_timestamp - latest_timestamp) / pd.Timedelta(hours=1))
    current_timestamp = latest_timestamp
    prediction = None

    for step in range(1, hours_ahead + 1):
        current_timestamp = latest_timestamp + pd.Timedelta(hours=step)
        is_final_step = current_timestamp == forecast_timestamp

        future_row = pd.DataFrame(
            {
                "temp": [temp],
                "hum": [hum],
                "windspeed": [windspeed],
                "weathersit": [weathersit],
                "workingday": [workingday],
                "yr": [current_timestamp.year - 2011],
                target_col: [np.nan],
            },
            index=[current_timestamp],
        )

        prediction_data = pd.concat([prediction_data, future_row])
        prediction_data = (
            prediction_data
            .loc[~prediction_data.index.duplicated(keep="last")]
            .sort_index()
        )

        engineered = create_tree_features(prediction_data)
        feature_row = engineered[TREE_FEATURES].iloc[-1:].copy()
        feature_row = feature_row.fillna(0.0)

        step_prediction = predictor.predict(feature_row)
        prediction_data.loc[current_timestamp, target_col] = step_prediction

        if is_final_step:
            prediction = step_prediction

    try:
        st.success("Prediction completed.")
        st.metric("Predicted Bike Demand", f"{prediction:,.0f}")
        st.write(f"Forecast timestamp: {forecast_timestamp}")
        if hours_ahead > 1:
            st.caption(
                f"Forecast is {hours_ahead} hour(s) ahead of the last known data point "
                f"and was produced recursively, one hour at a time."
            )
    except Exception as error:
        st.error("Prediction failed.")
        st.code(str(error))

st.write("historical data ---------------------------------------------------------------")
with st.expander("View recent historical data"):
    st.dataframe(historical_data.tail(20), use_container_width=True)
