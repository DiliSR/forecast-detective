"""Local explanations for one forecast hour.

LightGBM: SHAP (exact TreeSHAP) and LIME (repeated with several seeds to expose its instability).
LSTM: Integrated Gradients (Captum) and TimeSHAP (which past hours, which features, which cells).

Every method returns rows in one tidy format so they can be compared:
    method | feature | hours_before_target | timestamp | value | attribution_kwh
`hours_before_target` and `timestamp` locate a timestep for the sequence methods; they are empty
for tabular features. Attributions are in kWh: how much that input moved this forecast away from
the method's reference point. The reference points differ, which matters when comparing them:
- SHAP: the model's average forecast over all training hours (day and night). TreeSHAP with a
  custom background (e.g. "a normal 18:00") isn't available for LightGBM's categorical splits here.
- IG / TimeSHAP: a typical non-holiday weekday week ending at the same hour, same season.
- LIME: no reference; weights are slopes of a local linear surrogate, not a split of the forecast.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from captum.attr import IntegratedGradients
from lime.lime_tabular import LimeTabularExplainer
from timeshap.explainer import local_cell_level, local_event, local_feat, local_pruning

from forecast_explainer import PROJECT_ROOT
from forecast_explainer import lgbm_model as lm
from forecast_explainer import lstm_model as sm
from forecast_explainer.data import load_series
from forecast_explainer.features import CATEGORICAL_FEATURES, MIN_LAG

COLUMNS = ["method", "feature", "hours_before_target", "timestamp", "value", "attribution_kwh"]
EXPLANATIONS_DIR = PROJECT_ROOT / "outputs" / "explanations"


# ----------------------------------------------------------------------------- LightGBM

def lgbm_shap(booster, X_row: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Exact TreeSHAP values for one row. Returns (attributions, expected value of the model)."""
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(X_row)[0]
    rows = pd.DataFrame({
        "method": "shap",
        "feature": X_row.columns,
        "value": X_row.iloc[0].astype(float).to_numpy(),
        "attribution_kwh": values,
    })
    return _tidy(rows), float(explainer.expected_value)


def lgbm_lime(booster, X_train: pd.DataFrame, X_row: pd.DataFrame, n_repeats: int = 5,
              num_samples: int = 5000) -> tuple[pd.DataFrame, dict]:
    """LIME weights averaged over `n_repeats` seeds, with their spread.

    Extra columns: `attribution_std` (spread across seeds), `rank_min`/`rank_max` (how far the
    feature's importance rank moves between seeds), and `condition` (LIME's bin, e.g.
    "load_lag_168h > 230000"). The fit dict reports the surrogate's R^2 per seed: a low R^2 means
    the linear surrogate doesn't describe the model locally, so its weights mean little.
    """
    columns = list(X_row.columns)
    categorical = [i for i, c in enumerate(columns) if c in CATEGORICAL_FEATURES or X_train[c].nunique() <= 2]

    def predict(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=columns)
        for col in CATEGORICAL_FEATURES:
            frame[col] = frame[col].round().astype(int)
        return lm.predict(booster, frame)

    train_values = _as_float(X_train)
    row_values = _as_float(X_row)[0]
    runs, conditions, scores = [], {}, []
    for seed in range(n_repeats):
        explainer = LimeTabularExplainer(train_values, mode="regression", feature_names=columns,
                                         categorical_features=categorical, random_state=seed)
        exp = explainer.explain_instance(row_values, predict, num_features=len(columns), num_samples=num_samples)
        weights = dict(exp.local_exp[1])
        runs.append([weights.get(i, 0.0) for i in range(len(columns))])
        conditions.update({columns[i]: text for (text, _), (i, _) in zip(exp.as_list(), exp.local_exp[1])})
        scores.append(float(exp.score))

    runs = np.array(runs)
    ranks = (-np.abs(runs)).argsort(axis=1).argsort(axis=1) + 1
    rows = pd.DataFrame({
        "method": "lime",
        "feature": columns,
        "value": row_values,
        "attribution_kwh": runs.mean(axis=0),
        "attribution_std": runs.std(axis=0),
        "rank_min": ranks.min(axis=0),
        "rank_max": ranks.max(axis=0),
        "condition": [conditions.get(c, "") for c in columns],
    })
    fit = {"surrogate_r2_per_seed": scores, "n_repeats": n_repeats, "num_samples": num_samples}
    return _tidy(rows), fit


def _as_float(X: pd.DataFrame) -> np.ndarray:
    # Categorical columns hold their real values (hour 0-23, month 1-12), not category codes.
    return X.astype({c: int for c in CATEGORICAL_FEATURES}).astype(float).to_numpy()


# ----------------------------------------------------------------------------- LSTM

FLAG_FEATURES = ["is_holiday", "is_major_holiday", "target_is_weekend", "target_is_holiday", "target_is_major_holiday"]


def lstm_baseline(train_load: pd.Series, scaler: sm.Scaler, timestamp: pd.Timestamp,
                  season_window: int | None = 1) -> np.ndarray:
    """The reference input for IG and TimeSHAP, shape (SEQ_LEN, n_features).

    The average of training sequences whose target is the same hour of day on an ordinary
    (non-holiday) weekday in the same season (target month +- `season_window` months), with every
    holiday/weekend flag off: "a typical weekday at this hour, this time of year". Attributions then
    answer "why is this forecast different from a normal December 18:00?".
    - Hour-of-day features match the explained sequence, so they get ~zero attribution by design;
      an all-zeros or all-means baseline would credit them for "being an hour at all".
    - Without the season filter (`season_window=None`) the baseline includes the summer peak, and
      "December is below the yearly average" gets mixed into the load attribution.
    """
    X, _, index = sm.build_sequences(train_load, scaler)
    target_flags = X[:, -1, [sm.FEATURES.index(f) for f in ("target_is_weekend", "target_is_holiday")]]
    typical = (index.hour == timestamp.hour) & (target_flags.sum(axis=1) == 0)
    if season_window is not None:
        months = [(timestamp.month - 1 + d) % 12 + 1 for d in range(-season_window, season_window + 1)]
        typical &= index.month.isin(months)
    baseline = X[typical].mean(axis=0)
    baseline[:, [sm.FEATURES.index(f) for f in FLAG_FEATURES]] = 0.0
    return baseline.astype("float32")


def lstm_integrated_gradients(model, sequence: np.ndarray, baseline: np.ndarray, steps: pd.DatetimeIndex,
                              scaler: sm.Scaler, n_steps: int = 64) -> tuple[pd.DataFrame, float]:
    """IG attribution for every (timestep, feature) cell. Returns (cells, convergence delta in kWh).

    Completeness: the cells sum to forecast - baseline forecast; the delta is how far the
    numerical integral is from that (should be small relative to the attributions).
    """
    x = torch.from_numpy(sequence)
    baseline = torch.from_numpy(baseline[None])
    ig = IntegratedGradients(model)
    attr, delta = ig.attribute(x, baselines=baseline, target=0, n_steps=n_steps, return_convergence_delta=True)
    attr = attr[0].detach().numpy() * scaler.std

    n_steps_seq, n_feats = attr.shape
    hours_before = MIN_LAG + np.arange(n_steps_seq)[::-1]
    values = sequence[0].copy()
    values[:, 0] = scaler.unscale(values[:, 0])  # report load in kWh
    rows = pd.DataFrame({
        "method": "integrated_gradients",
        "feature": np.tile(sm.FEATURES, n_steps_seq),
        "hours_before_target": np.repeat(hours_before, n_feats),
        "timestamp": np.repeat(steps, n_feats),
        "value": values.reshape(-1),
        "attribution_kwh": attr.reshape(-1),
    })
    return _tidy(rows), float(delta.item() * scaler.std)


def ig_by_feature(cells: pd.DataFrame) -> pd.DataFrame:
    """Collapse IG cells over time: total attribution per input feature."""
    return cells.groupby("feature", sort=False)["attribution_kwh"].sum().reindex(sm.FEATURES).reset_index()


def ig_by_timestep(cells: pd.DataFrame) -> pd.DataFrame:
    """Collapse IG cells over the past hour's own features: total attribution per past hour.

    The target-hour flags are repeated on every step, so they are excluded here (they describe the
    hour being forecast, not the past hour); `ig_by_feature` reports them.
    """
    own = cells[cells["feature"].isin(sm.STEP_FEATURES)]
    return own.groupby(["hours_before_target", "timestamp"])["attribution_kwh"].sum().reset_index()


def lstm_timeshap(model, sequence: np.ndarray, baseline: np.ndarray, steps: pd.DatetimeIndex,
                  scaler: sm.Scaler, pruning_tol_kwh: float = 500.0, nsamples: int = 2000,
                  top_cells: int = 4, seed: int = 42) -> dict:
    """TimeSHAP at three levels: events (past hours), features, and cells (hour x feature).

    Pruning first groups the oldest hours whose joint effect is below `pruning_tol_kwh` into one
    "older hours (pruned)" event, which keeps the Shapley estimation tractable for 168 steps.
    An event carries every feature of that step, including the repeated target-hour flags, so a
    recent hour's event value also contains part of "the target is Christmas"; the cell level
    separates the two. Rows get an `event` column: "t-24h", "older hours (pruned)" or "other hours".
    """
    def f(x: np.ndarray) -> np.ndarray:
        return sm.predict_scaled(model, x)[:, None]

    tol = pruning_tol_kwh / scaler.std
    _, prune_idx = local_pruning(f, sequence, {"tol": tol}, baseline)
    pruned_idx = sequence.shape[1] + prune_idx  # first timestep kept as an individual event

    events = local_event(f, sequence, {"rs": seed, "nsamples": nsamples}, None, None, baseline, pruned_idx)
    feats = local_feat(f, sequence, {"rs": seed, "nsamples": nsamples, "feature_names": sm.FEATURES},
                       None, None, baseline, pruned_idx)
    cells = local_cell_level(f, sequence, {"rs": seed, "nsamples": nsamples, "top_x_events": top_cells,
                                           "top_x_feats": top_cells}, events, feats, None, None,
                             baseline, pruned_idx)
    return {
        "events": _timeshap_rows(events, steps, scaler, "timeshap_event"),
        "features": _timeshap_rows(feats, steps, scaler, "timeshap_feature"),
        "cells": _timeshap_rows(cells, steps, scaler, "timeshap_cell"),
        "n_events_kept": int(sequence.shape[1] - pruned_idx),
    }


TIMESHAP_GROUPS = {"Pruned Events": "older hours (pruned)", "Other Events": "other hours",
                   "Other Features": "(other features)"}


def _timeshap_rows(df: pd.DataFrame, steps: pd.DatetimeIndex, scaler: sm.Scaler, method: str) -> pd.DataFrame:
    """Convert TimeSHAP's frames (labels like "Event -3", "Pruned Events") to the tidy format."""
    if method == "timeshap_event":
        events, features = df["Feature"], ["(all features)"] * len(df)
    elif method == "timeshap_feature":
        events, features = [None] * len(df), df["Feature"].replace(TIMESHAP_GROUPS)
        features = features.replace({"older hours (pruned)": "(older hours, pruned)"})
    else:
        events, features = df["Event"], df["Feature"].replace(TIMESHAP_GROUPS)

    hours, stamps, labels = [], [], []
    for label in events:
        if isinstance(label, str) and label.startswith("Event "):
            k = -int(label.split()[1])  # "Event -1" is the most recent step, i.e. t-24h
            hours.append(MIN_LAG + k - 1); stamps.append(steps[-k]); labels.append(f"t-{MIN_LAG + k - 1}h")
        else:
            hours.append(np.nan); stamps.append(pd.NaT); labels.append(TIMESHAP_GROUPS.get(label))
    rows = pd.DataFrame({
        "method": method,
        "feature": list(features),
        "hours_before_target": hours,
        "timestamp": stamps,
        "value": np.nan,
        "attribution_kwh": df["Shapley Value"].to_numpy() * scaler.std,
        "event": labels,
    })
    return _tidy(rows)


def _tidy(rows: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMNS:
        if col not in rows:
            rows[col] = np.nan if col != "timestamp" else pd.NaT
    extra = [c for c in rows.columns if c not in COLUMNS]
    rows = rows[COLUMNS + extra]
    return rows.reindex(rows["attribution_kwh"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ----------------------------------------------------------------------------- one forecast

@dataclass
class ForecastExplanation:
    timestamp: pd.Timestamp
    client: str
    actual_kwh: float
    lgbm: dict = field(default_factory=dict)
    lstm: dict = field(default_factory=dict)

    def tables(self) -> dict[str, pd.DataFrame]:
        out = {k: v for k, v in self.lgbm.items() if isinstance(v, pd.DataFrame)}
        out |= {k: v for k, v in self.lstm.items() if isinstance(v, pd.DataFrame)}
        return out

    def to_dict(self) -> dict:
        def clean(obj):
            if isinstance(obj, pd.DataFrame):
                return json.loads(obj.to_json(orient="records", date_format="iso"))
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            return obj
        return {"timestamp": str(self.timestamp), "client": self.client, "actual_kwh": self.actual_kwh,
                "lightgbm": clean(self.lgbm), "lstm": clean(self.lstm)}

    def save(self, directory: Path = EXPLANATIONS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.client.lower()}_{self.timestamp:%Y-%m-%d_%H%M}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


def explain_forecast(timestamp, client: str = "total", lime_repeats: int = 5, timeshap_nsamples: int = 2000,
                     verbose: bool = True) -> ForecastExplanation:
    """Run all four explainers for one target hour, using the saved models for `client`."""
    timestamp = pd.Timestamp(timestamp)
    log = print if verbose else (lambda *a, **k: None)
    load = load_series(client)
    name = client.lower()

    # LightGBM
    booster, meta = lm.load(f"lgbm_{name}")
    X, _ = lm.build_dataset(load)
    X = X[meta["features"]]
    X_row = X.loc[[timestamp]]
    X_train = X.loc[: meta["split"]["train_end"]]
    explanation = ForecastExplanation(timestamp, client, float(load.loc[timestamp]))

    log("LightGBM: SHAP")
    shap_rows, expected = lgbm_shap(booster, X_row)
    log(f"LightGBM: LIME x{lime_repeats} seeds")
    lime_rows, lime_fit = lgbm_lime(booster, X_train, X_row, n_repeats=lime_repeats)
    explanation.lgbm = {"prediction_kwh": float(lm.predict(booster, X_row)[0]), "expected_value_kwh": expected,
                        "shap": shap_rows, "lime": lime_rows, "lime_fit": lime_fit}

    # LSTM
    model, scaler, lstm_meta = sm.load(f"lstm_{name}")
    sequence, steps = sm.sequence_for(load, scaler, timestamp)
    baseline = lstm_baseline(load.loc[: lstm_meta["split"]["train_end"]], scaler, timestamp)
    prediction = float(scaler.unscale(sm.predict_scaled(model, sequence)[0]))
    baseline_prediction = float(scaler.unscale(sm.predict_scaled(model, baseline[None])[0]))

    log("LSTM: Integrated Gradients")
    ig_cells, ig_delta = lstm_integrated_gradients(model, sequence, baseline, steps, scaler)
    log("LSTM: TimeSHAP (pruning, events, features, cells)")
    ts = lstm_timeshap(model, sequence, baseline, steps, scaler, nsamples=timeshap_nsamples)
    explanation.lstm = {
        "prediction_kwh": prediction, "baseline_prediction_kwh": baseline_prediction,
        "ig_cells": ig_cells, "ig_by_feature": ig_by_feature(ig_cells), "ig_by_timestep": ig_by_timestep(ig_cells),
        "ig_convergence_delta_kwh": ig_delta,
        "timeshap_events": ts["events"], "timeshap_features": ts["features"], "timeshap_cells": ts["cells"],
        "timeshap_events_kept": ts["n_events_kept"],
    }
    return explanation
