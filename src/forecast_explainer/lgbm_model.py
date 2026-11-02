"""LightGBM day-ahead load forecaster: train, evaluate against baselines, save and load."""

import json
from dataclasses import asdict, dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecast_explainer import MODELS_DIR
from forecast_explainer.features import encode_categoricals, make_features


@dataclass
class Split:
    train_end: str = "2013-12-31 23:00"
    valid_end: str = "2014-06-30 23:00"  # valid drives early stopping; test is everything after


@dataclass
class LgbmParams:
    objective: str = "regression"
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_child_samples: int = 20  # small enough for 2 training Christmases (48 rows) to get a leaf
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    seed: int = 42
    verbose: int = -1
    extra: dict = field(default_factory=dict)

    def to_lgb(self) -> dict:
        params = {k: v for k, v in asdict(self).items() if k != "extra"}
        return params | self.extra


def build_dataset(load: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    X = make_features(load)
    rows = X.notna().all(axis=1) & load.notna()
    return X[rows], load[rows]


def split_frame(X: pd.DataFrame, y: pd.Series, split: Split) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    train = X.index <= split.train_end
    valid = (X.index > split.train_end) & (X.index <= split.valid_end)
    test = X.index > split.valid_end
    return {name: (X[mask], y[mask]) for name, mask in [("train", train), ("valid", valid), ("test", test)]}


def train(X_train, y_train, X_valid, y_valid, params: LgbmParams = LgbmParams(),
          num_boost_round: int = 5000, early_stopping: int = 200) -> lgb.Booster:
    return lgb.train(
        params.to_lgb(),
        lgb.Dataset(X_train, y_train),
        num_boost_round=num_boost_round,
        valid_sets=[lgb.Dataset(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(0)],
    )


def metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape_pct": float(100 * np.mean(np.abs(err) / np.asarray(y_true))),
    }


def baselines(X: pd.DataFrame) -> dict[str, pd.Series]:
    """Naive forecasts the model has to beat to be worth explaining."""
    return {
        "naive_yesterday": X["load_lag_24h"],
        "naive_last_week": X["load_lag_168h"],
        "same_hour_mean_4w": X["load_same_hour_mean_4w"],
    }


def save(booster: lgb.Booster, name: str, meta: dict, predictions: pd.DataFrame) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(MODELS_DIR / f"{name}.txt")
    (MODELS_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))
    predictions.to_parquet(MODELS_DIR / f"{name}_predictions.parquet")


def load(name: str) -> tuple[lgb.Booster, dict]:
    booster = lgb.Booster(model_file=MODELS_DIR / f"{name}.txt")
    meta = json.loads((MODELS_DIR / f"{name}.json").read_text())
    return booster, meta


def load_predictions(name: str) -> pd.DataFrame:
    return encode_categoricals(pd.read_parquet(MODELS_DIR / f"{name}_predictions.parquet"))


def predict(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(encode_categoricals(X[booster.feature_name()]))
