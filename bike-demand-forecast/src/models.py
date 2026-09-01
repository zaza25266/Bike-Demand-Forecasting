import tensorflow as tf

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from src.config import RANDOM_STATE


def build_lightgbm():
    """
    Build the LightGBM regression model.
    """

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return model


def build_random_forest():
    """
    Build the Random Forest regression model.
    """

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return model


def build_lstm(
    window=168,
    n_features=12
):
    """
    Build the LSTM forecasting model.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(window, n_features)
        ),

        tf.keras.layers.LSTM(
            64,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(
            32
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model