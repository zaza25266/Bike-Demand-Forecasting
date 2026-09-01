#  Bike Demand Forecasting

An hourly bike-demand forecasting project comparing **classical time-series forecasting, machine learning, and deep learning approaches**.

The project focuses on data preprocessing, exploratory analysis, statistical analysis, temporal feature engineering, model training, evaluation, model comparison, experiment tracking, automated testing, and deployment through a Streamlit prediction application.

---

##  Project Overview

The goal is to forecast **hourly bike rental demand** using historical demand, weather, calendar, and temporal features.

The project compares:

* Classical time-series models
* Seasonal baseline models
* Tree-based machine-learning models
* Deep-learning sequence models

The overall workflow is:

```text
Raw Data
   ↓
Data Preprocessing
   ↓
EDA & Statistical Analysis
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model
   ↓
Streamlit Prediction Application
```

---

##  Dataset

The project uses the **Bike Sharing Dataset** containing hourly bike rental records.

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

---

#  Models

## Classical Time-Series Models

The following classical forecasting approaches were evaluated:

* Naive
* Daily Seasonal Naive
* Weekly Seasonal Naive
* ARIMA
* Holt-Winters
* SARIMAX

## Machine Learning Models

Tree-based models were trained using temporal and historical-demand features:

* Random Forest
* XGBoost
* LightGBM

## Deep Learning

A sequence-based:

* LSTM

was trained to learn temporal dependencies from historical observations and external variables.

---

#  Model Performance

Models were evaluated on the test set using:

* **MAE — Mean Absolute Error**
* **RMSE — Root Mean Squared Error**

Lower values indicate better performance.

| Rank | Model                  |       MAE |      RMSE |
| ---: | ---------------------- | --------: | --------: |
|    1 | **LightGBM**           | **33.09** | **51.20** |
|    2 | **LSTM**               | **33.51** | **51.69** |
|    3 | Random Forest          |     40.88 |     64.80 |
|    4 | XGBoost                |     42.33 |     69.20 |
|    5 | Weekly Seasonal Naive  |     67.95 |    113.95 |
|    6 | Daily Seasonal Naive   |    128.78 |    184.82 |
|    7 | ARIMA                  |    273.21 |    313.46 |
|    8 | Naive                  |    281.42 |    322.13 |
|    9 | SARIMAX (24h, 10 iter) |    395.25 |    437.32 |
|   10 | Holt-Winters           |    685.31 |    779.45 |

---

##  Best Model

**LightGBM** achieved the best test-set performance:

```text
MAE  = 33.09
RMSE = 51.20
```

**LSTM** was a close second:

```text
MAE  = 33.51
RMSE = 51.69
```

Therefore, **LightGBM was selected as the prediction model** based on the evaluated test-set metrics.

The small difference between LightGBM and LSTM also shows that a more complex deep-learning model does not automatically provide better forecasting performance.

---

#  Feature Engineering

The machine-learning pipeline creates temporal and historical-demand features.

## Calendar Features

* `hour`
* `day_of_week`
* `month`
* `day_of_year`
* `week_of_year`
* `is_weekend`

## Cyclical Features

Daily and weekly periodicity is represented using:

```text
hour_sin
hour_cos

dow_sin
dow_cos
```

Cyclical encoding allows relationships such as:

```text
23 → 0
```

to be represented naturally instead of treating the values as unrelated numerical categories.

---

## Lag Features

Historical demand is used to generate:

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

For example:

```text
lag_24
```

represents demand from approximately 24 hours earlier.

```text
lag_168
```

represents demand from the same hour approximately one week earlier.

---

## Rolling Features

Rolling demand statistics are calculated using:

```text
rolling_mean_24
rolling_std_24

rolling_mean_168
rolling_std_168
```

The rolling calculations are shifted so that the current target value is not used as an input feature, preventing target leakage.

---

#  LSTM

The LSTM uses a sequence-based representation of historical observations.

Its input features include:

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

---

#  Data Processing

The preprocessing pipeline performs the following steps:

1. Load the raw hourly dataset.
2. Convert the datetime column into a datetime index.
3. Sort observations chronologically.
4. Restore the complete hourly frequency.
5. Identify missing observations.
6. Interpolate continuous variables.
7. Forward/backward fill appropriate discrete variables.
8. Save the processed dataset.

The processed dataset is then used by the forecasting pipelines.

---

#  Project Structure

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

> **Note:** Trained model files are not stored in this GitHub repository because some model artifacts are too large for GitHub's file-size limits.

---

#  Model Storage

The trained production model is stored on **Hugging Face** rather than inside the GitHub repository.

This keeps the GitHub repository lightweight while allowing the Streamlit application to retrieve the required model artifacts when running.

The prediction flow is:

```text
Streamlit
    ↓
predictor.py
    ↓
Hugging Face Hub
    ↓
Download trained model
    ↓
Load model
    ↓
Generate prediction
    ↓
Display result
```

The application does **not** require the trained model to be committed to GitHub.

Instead, `predictor.py` handles downloading/loading the required production artifacts from Hugging Face.

---

#  Automated Pipeline

The project provides a single command for running the training and evaluation workflow:

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
5. Compare model performance
```

---

#  MLflow Experiment Tracking

**MLflow** is used during model development to track experiments.

Tracked information includes:

* Model parameters
* Evaluation metrics
* Experiment runs
* Artifacts

The local MLflow tracking directory is excluded from GitHub because it contains generated experiment files.

---

#  Testing

The project uses **Pytest** for automated testing.

Run the tests with:

```bash
python3 -m pytest
```

The test suite validates core feature-engineering and pipeline functionality.

---

#  Streamlit Application

A Streamlit application provides an interface for generating bike-demand predictions.

Start the application with:

```bash
python3 -m streamlit run app.py
```

The application uses `predictor.py` to retrieve the trained production model from Hugging Face and generate predictions.

```text
User Input
    ↓
Streamlit UI
    ↓
predictor.py
    ↓
Hugging Face Model
    ↓
Prediction
    ↓
Streamlit UI
```

---

#  Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd bike-demand-forecast
```

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Make the Pipeline Executable

```bash
chmod +x run.sh
```

## 5. Run the Training Pipeline

```bash
./run.sh
```

## 6. Start the Streamlit Application

```bash
python3 -m streamlit run app.py
```

---

#  Evaluation Strategy

Because this is a **time-series forecasting problem**, chronological ordering is preserved during dataset splitting.

The project uses separate:

* Training period
* Validation period
* Test period

rather than randomly shuffling observations.

The final models are compared using:

### MAE

**Mean Absolute Error** measures the average absolute difference between predictions and actual values.

### RMSE

**Root Mean Squared Error** gives greater weight to larger prediction errors.

---

#  Results and Analysis

The results show that the machine-learning and deep-learning approaches substantially outperform the simpler classical forecasting approaches on this dataset.

The strongest results were:

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

LightGBM achieved the lowest MAE and RMSE among the evaluated models.

The close performance between LightGBM and LSTM demonstrates that increasing model complexity does not automatically produce better forecasting performance.

---

#  Key Takeaways

1. **Seasonal baselines provide an important reference point.**
2. **Classical models such as ARIMA and Holt-Winters performed poorly compared with the feature-based approaches on this dataset.**
3. **Tree-based models performed strongly after temporal and lag feature engineering.**
4. **LightGBM achieved the best overall test performance.**
5. **LSTM achieved nearly identical performance to LightGBM.**
6. **Chronological evaluation is important for avoiding unrealistic random train/test splits in time-series forecasting.**
7. **The production model is stored externally on Hugging Face rather than committed to GitHub.**

---

#  Technologies

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM

### Time Series

* Statsmodels

### Deep Learning

* TensorFlow
* Keras

### Experiment Tracking

* MLflow

### Testing

* Pytest

### Application

* Streamlit

### Model Hosting

* Hugging Face

### Version Control

* Git
* GitHub
