"""Accuracy of every registered model, overall and in slices (hour of day, day type, month).

Why slices? One overall error figure hides where a model fails. Our own results show it: LightGBM is 3.0%
overall but ~28% off on Christmas. The planning agent needs the sliced view to answer "which model
is best *where*", and the explanation agent uses it to say "this hour is a known weak spot".

Error conventions used everywhere in this module:
    error      = forecast - actual                       (kWh)
    error_pct  = 100 * error / actual                    (positive = over-forecast)
    RMSE       = sqrt(mean(error^2))                     (kWh; the headline measure, big misses weigh more)
    MAPE       = mean(|error_pct|)                       (size of the miss, ignores direction)
    bias_pct   = mean(error_pct)                         (direction: >0 means the model runs high)
"""

import numpy as np
import pandas as pd

from forecast_explainer import lgbm_model as lm
from forecast_explainer.features import calendar_features
from forecast_explainer.registry import MODELS, all_predictions

SLICES = ["overall", "part_of_day", "day_type", "month"]


def prediction_frame(client: str = "total", models: list[str] | None = None) -> pd.DataFrame:
    """Test-set forecasts for all models plus the calendar columns used for slicing.

    Index = target hour. Columns: y_true, one column per model, then hour / day_of_week / month /
    is_weekend / is_holiday / is_major_holiday / is_bridge_day from features.calendar_features
    (the same calendar the models were trained with, so "holiday" means the same thing here).
    """
    preds = all_predictions(client, models)
    return preds.join(calendar_features(preds.index))


def _slice_labels(frame: pd.DataFrame, by: str) -> pd.Series:
    """Label every hour with the group it belongs to for the chosen slicing.

    by="overall"     one group, all hours.
    by="part_of_day" night 00-05 / morning 06-11 / afternoon 12-17 / evening 18-23. Error differs
                     by time of day (night is easy and flat, afternoons are volatile).
    by="day_type"    major holiday (25 Dec, 1 Jan) / holiday / weekend / weekday. np.select takes
                     the first matching condition, so the most specific label wins.
    by="month"       calendar month name, to spot seasonal drift.
    """
    if by == "overall":
        return pd.Series("all hours", index=frame.index)
    if by == "part_of_day":
        bins = pd.cut(frame["hour"], [-1, 5, 11, 17, 23],
                      labels=["night 00-05", "morning 06-11", "afternoon 12-17", "evening 18-23"])
        return bins.astype(str)
    if by == "day_type":
        return pd.Series(np.select(
            [frame["is_major_holiday"] == 1, frame["is_holiday"] == 1, frame["is_weekend"] == 1],
            ["major holiday", "holiday", "weekend"], default="weekday"), index=frame.index)
    if by == "month":
        return frame.index.strftime("%m %b").to_series(index=frame.index)
    raise ValueError(f"unknown slice {by!r}; choose from {SLICES}")


def accuracy_by(by: str = "overall", client: str = "total", models: list[str] | None = None) -> pd.DataFrame:
    """Accuracy per model and per group, in a tidy table (one row per slice x model).

    by:     how to group the hours - one of SLICES.
    models: restrict to some model keys (default all).

    Columns: slice, model, n (hours), mae and rmse (kWh), mape_pct, bias_pct.
    Tidy ("long") format on purpose: it is easy to filter, plot, and hand to an LLM as JSON.
    The error definitions come from lgbm_model.metrics so there is exactly one definition of
    MAPE/MAE/RMSE in the project; only bias is added here.
    """
    frame = prediction_frame(client, models)
    labels = _slice_labels(frame, by)
    rows = []
    for model in models or list(MODELS):
        for label, group in frame.groupby(labels):
            score = lm.metrics(group["y_true"], group[model])
            score["bias_pct"] = float(100 * ((group[model] - group["y_true"]) / group["y_true"]).mean())
            rows.append({"slice": label, "model": model, "n": len(group), **score})
    return pd.DataFrame(rows)


def error_frame(client: str = "total") -> pd.DataFrame:
    """Per-hour errors: y_true, forecasts, `<model>_err_pct`, and `model_gap_pct` between models.

    model_gap_pct = |LightGBM - LSTM| as a % of their average. It needs no ground truth, so it is
    also available for future hours - it is the cheapest "do the models agree?" signal we have.
    """
    frame = prediction_frame(client)
    out = frame[["y_true", *MODELS]].copy()
    for model in MODELS:
        out[f"{model}_err_pct"] = 100 * (frame[model] - frame["y_true"]) / frame["y_true"]
    first, second = list(MODELS)[:2]
    out["model_gap_pct"] = 100 * (frame[first] - frame[second]).abs() / frame[[first, second]].mean(axis=1)
    return out


def interesting_hours(client: str = "total", n: int = 3) -> pd.DataFrame:
    """Hours worth explaining: each model's worst misses and the largest disagreement between models.

    n: how many hours per category.

    Why one per day? The worst hours cluster (Christmas Eve 17:00, 18:00 and 19:00 are all bad for
    the same reason). Keeping only the single worst hour of each day gives n *different*
    situations, which is what a planner or an evaluation set needs.
    Columns: kind, y_true, <models>, <model>_err_pct, model_gap_pct. Index = target hour.
    """
    errors = error_frame(client)
    rankings = {f"worst_{m}": errors[f"{m}_err_pct"].abs() for m in MODELS}
    rankings["largest_model_gap"] = errors["model_gap_pct"]
    picked = []
    for kind, score in rankings.items():
        best_per_day = score.groupby(score.index.normalize()).idxmax()  # one hour per calendar day
        top = score.loc[best_per_day].nlargest(n).index
        picked.append(errors.loc[top].assign(kind=kind))
    return pd.concat(picked).set_index("kind", append=True).reset_index(level="kind").round(2)
