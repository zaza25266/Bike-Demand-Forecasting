# Bike Demand Forecasting

An end-to-end hourly bike-demand forecasting project comparing classical time-series forecasting, machine learning, and deep learning approaches.

The project covers data preprocessing, exploratory analysis, statistical analysis, feature engineering, model training, evaluation, MLflow experiment tracking, automated testing, model selection, and a Streamlit prediction application.

## Project Overview

The goal is to forecast hourly bike rental demand using historical demand, weather, calendar, and temporal features.

The project compares:

* Classical time-series models
* Tree-based machine-learning models
* Deep-learning sequence models
* Seasonal baseline models

The complete workflow is automated through `run.sh`.

## Dataset

The project uses the **Bike Sharing Dataset** with hourly rental records.

The prediction target is:

```text
cnt
```

where `cnt` represents the total number of bike rentals during an hour.

Important variables include:

* Temperature
* Humidity
* Windspeed
* Weather situation
* Holiday
* Working day
* Year
* Hour
* Date/time

## Models

### Classical Time Series

* Naive
* Daily Seasonal Naive
* Weekly Seasonal Naive
* ARIMA
* Holt-Winters
* SARIMAX

### Machine Learning

* Random Forest
* XGBoost
* LightGBM

### Deep Learning

* LSTM

## Model Performance

Models were evaluated on the test set using **MAE** and **RMSE**.

Lower values indicate better performance.

| Rank | Model                  |       MAE |      RMSE |
| ---: | ---------------------- | --------: | --------: |
| 🥇 1 | **LightGBM**           | **33.09** | **51.20** |
| 🥈 2 | **LSTM**               | **33.51** | **51.69** |
|    3 | Random Forest          |     40.88 |     64.80 |
|    4 | XGBoost                |     42.33 |     69.20 |
|    5 | Weekly Seasonal Naive  |     67.95 |    113.95 |
|    6 | Daily Seasonal Naive   |    128.78 |    184.82 |
|    7 | ARIMA                  |    273.21 |    313.46 |
|    8 | Naive                  |    281.42 |    322.13 |
|    9 | SARIMAX (24h, 10 iter) |    395.25 |    437.32 |
|   10 | Holt-Winters           |    685.31 |    779.45 |

### Best Model

LightGBM achieved the best test-set performance:

```text
MAE  = 33.09
RMSE = 51.20
```

LSTM was a close second:

```text
MAE  = 33.51
RMSE = 51.69
```

Therefore, **LightGBM was selected as the best-performing model** based on the test-set metrics.

## Feature Engineering

The machine-learning pipeline creates temporal and historical demand features.

### Calendar Features

* Hour
* Day of week
* Month
* Day of year
* Week of year
* Weekend indicator

### Cyclical Features

Daily and weekly periodicity is represented using:

* `hour_sin`
* `hour_cos`
* `dow_sin`
* `dow_cos`

This allows cyclical relationships such as `23 → 0` to be represented more naturally.

### Lag Features

Historical demand is used to create:

```text
lag_1
lag_2
lag_3
lag_6
lag_12
lag_24
lag_48
lag_72
lag_168
```

The `lag_168` feature represents demand from the same hour one week earlier.

### Rolling Features

Historical demand is also used to calculate:

```text
rolling_mean_24
rolling_std_24
rolling_mean_168
rolling_std_168
```

The rolling calculations are shifted so that the current target value is not used as an input feature.

## LSTM

The LSTM uses a sequence-based representation of historical observations.

Its feature set includes:

* Historical demand
* Temperature
* Humidity
* Windspeed
* Weather situation
* Holiday
* Working day
* Year
* Hour cyclical features
* Day-of-week cyclical features

The sequence model learns temporal dependencies from historical observations.

## Data Processing

The preprocessing pipeline:

1. Loads the raw hourly dataset.
2. Converts the datetime column into a datetime index.
3. Sorts observations chronologically.
4. Restores the complete hourly frequency.
5. Handles missing observations.
6. Interpolates continuous variables.
7. Forward/backward fills appropriate discrete variables.
8. Saves the processed dataset.

The resulting processed data is used by the downstream forecasting pipelines.

## Project Structure

```text
bike-demand-forecast/
│
├── app.py
├── requirements.txt
├── run.sh
├── .gitignore
├── README.md
│
├── data/
│   ├── day.csv
│   ├── hour.csv
│   └── hour_cleaned.csv
│
├── models/
│   ├── evaluation_results.csv
│   ├── features.joblib
│   ├── lightgbm.joblib
│   ├── lstm.keras
│   ├── metadata.json
│   ├── random_forest.joblib
│   └── scaler.joblib
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda_and_visualization.ipynb
│   ├── 03_statistical_analysis.ipynb
│   ├── 04_Classical_Time-Series_Forecasting.ipynb
│   ├── 05_MachineLearning_Forecasting.ipynb
│   └── 06_Deep_Learning_Forecasting.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── model_selection.py
│   ├── models.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
└── tests/
    └── test_pipeline.py
```

## Automated Pipeline

The project provides a single command for running the pipeline:

```bash
./run.sh
```

The pipeline performs:

```text
1. Run automated tests
        ↓
2. Prepare and process data
        ↓
3. Train forecasting models
        ↓
4. Evaluate models
        ↓
5. Select the best model
```

The selected model is stored in:

```text
models/metadata.json
```

## MLflow

MLflow is used for experiment tracking during model development.

The experiment records model runs and associated parameters, metrics, and artifacts.

The local MLflow tracking directory is intentionally excluded from GitHub because it contains generated experiment artifacts.

## Testing

The project uses Pytest for automated testing.

Run all tests:

```bash
python3 -m pytest
```

The test suite validates core feature-engineering and pipeline functionality.

## Streamlit Application

A Streamlit application is provided for interacting with the trained forecasting system.

Start the application with:

```bash
python3 -m streamlit run app.py
```

The application loads the trained production artifacts and provides a user interface for generating demand predictions.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd bike-demand-forecast
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make the pipeline executable:

```bash
chmod +x run.sh
```

Run the complete pipeline:

```bash
./run.sh
```

Start the application:

```bash
python3 -m streamlit run app.py
```

## Evaluation Strategy

Because this is a time-series forecasting problem, chronological ordering is preserved during dataset splitting.

The project uses separate training, validation, and test periods rather than randomly shuffling observations.

The final models are compared using:

* **MAE — Mean Absolute Error**
* **RMSE — Root Mean Squared Error**

MAE measures the average absolute prediction error, while RMSE gives greater weight to larger errors.

## Results and Analysis

The results show that the machine-learning and deep-learning approaches substantially outperform the simpler classical models on this dataset.

The strongest results were obtained by:

```text
LightGBM
MAE  = 33.09
RMSE = 51.20

LSTM
MAE  = 33.51
RMSE = 51.69

Random Forest
MAE  = 40.88
RMSE = 64.80

XGBoost
MAE  = 42.33
RMSE = 69.20
```

LightGBM achieved the lowest MAE and RMSE across the evaluated models.

The close performance of LightGBM and LSTM also shows that increasing model complexity does not automatically produce a better forecasting result.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Statsmodels
* XGBoost
* LightGBM
* TensorFlow / Keras
* MLflow
* Streamlit
* Pytest
* Git
* GitHub

## Key Takeaways

This project demonstrates an end-to-end approach to practical time-series forecasting rather than relying on a single algorithm.

The main findings were:

1. Seasonal baselines provide an important reference point.
2. Classical models such as ARIMA and Holt-Winters performed poorly compared with the stronger feature-based approaches on this dataset.
3. Tree-based models performed strongly after temporal and lag feature engineering.
4. LightGBM achieved the best overall test performance.
5. LSTM achieved nearly identical performance to LightGBM.
6. Automated testing, experiment tracking, model evaluation, and model selection were incorporated into the project pipeline.
