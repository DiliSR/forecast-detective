"""Unit tests for the disagreement checks, using tiny hand-made explanations (no models needed)."""

import pytest

from forecast_explainer import disagreement as dg


def table(**attrs):
    """A saved-table stand-in: {feature: kWh} -> list of records."""
    return [{"feature": f, "attribution_kwh": v} for f, v in attrs.items()]


def make_explanation(lg=200_000, ls=200_000, actual=200_000, shap=None, lime=None, r2=0.8):
    """A minimal explanation dict with the keys the checks read."""
    shap = shap or table(load_lag_24h=10_000, hour=5_000, is_holiday=0.0, month=1_000)
    lime = lime or table(load_lag_24h=9_000, hour=4_000, is_holiday=0.0, month=500)
    lime = [dict(r, rank_min=i + 1, rank_max=i + 1) for i, r in enumerate(lime)]
    return {
        "actual_kwh": actual,
        "lightgbm": {"prediction_kwh": lg, "shap": shap, "lime": lime,
                     "lime_fit": {"surrogate_r2_per_seed": [r2, r2, r2]}},
        "lstm": {"prediction_kwh": ls, "baseline_prediction_kwh": 190_000, "ig_convergence_delta_kwh": 10.0,
                 "ig_by_feature": table(load=9_000, hour_sin=100.0, hour_cos=100.0, target_is_holiday=0.0),
                 "timeshap_features": table(load=9_500, hour_sin=80.0, hour_cos=90.0, target_is_holiday=0.0),
                 "ig_by_timestep": [{"hours_before_target": 24, "attribution_kwh": 9_000}],
                 "timeshap_events": [{"hours_before_target": 24.0, "attribution_kwh": 9_500}]},
    }


def by_type(flags):
    return {f["type"]: f for f in flags}


def test_agreeing_models_give_ok():
    flags = by_type(dg.find_disagreements(make_explanation()))
    assert flags["model_gap"]["severity"] == "ok"
    assert flags["shared_error"]["severity"] == "ok"


def test_large_model_gap_is_high():
    flags = by_type(dg.find_disagreements(make_explanation(lg=200_000, ls=230_000)))
    assert flags["model_gap"]["severity"] == "high"
    assert flags["model_gap"]["numbers"]["gap_pct"] == pytest.approx(13.953, abs=0.01)


def test_both_models_over_forecast_is_shared_error():
    flags = by_type(dg.find_disagreements(make_explanation(lg=240_000, ls=235_000, actual=200_000)))
    assert flags["shared_error"]["severity"] == "high"
    assert "over-forecast" in flags["shared_error"]["message"]


def test_only_one_model_off_is_warning():
    flags = by_type(dg.find_disagreements(make_explanation(lg=240_000, ls=201_000, actual=200_000)))
    assert flags["shared_error"]["severity"] == "warning"


def test_sign_flip_between_shap_and_lime_is_detected():
    shap = table(load_lag_24h=10_000, hour=5_000, month=1_000, is_holiday=0.0)
    lime = table(load_lag_24h=-9_000, hour=4_000, month=500, is_holiday=0.0)  # flipped sign
    flags = by_type(dg.find_disagreements(make_explanation(shap=shap, lime=lime)))
    assert flags["shap_vs_lime"]["severity"] != "ok"
    assert "load_lag_24h" in flags["shap_vs_lime"]["numbers"]["sign_flips"]


def test_low_lime_r2_is_flagged():
    flags = by_type(dg.find_disagreements(make_explanation(r2=0.1)))
    assert flags["lime_fit"]["severity"] == "warning"


def test_results_sorted_most_severe_first():
    flags = dg.find_disagreements(make_explanation(lg=200_000, ls=230_000))
    order = [dg.SEVERITY_ORDER[f["severity"]] for f in flags]
    assert order == sorted(order)


def test_rank_correlation_ignores_immaterial_features():
    # Five features agree perfectly; the near-zero ones are in opposite order.
    import pandas as pd
    a = pd.Series({"a": 100.0, "b": 80.0, "c": 60.0, "d": 40.0, "x": 0.1, "y": 0.2})
    b = pd.Series({"a": 90.0, "b": 70.0, "c": 50.0, "d": 30.0, "x": 0.2, "y": 0.1})
    assert dg._rank_corr(a, b) < 1.0
    assert dg._rank_corr(a, b, min_abs=10.0) == pytest.approx(1.0)


def test_rank_correlation_not_applicable_with_few_features():
    import numpy as np
    import pandas as pd
    a = pd.Series({"a": 100.0, "x": 0.1, "y": 0.2, "z": 0.3})
    assert np.isnan(dg._rank_corr(a, a, min_abs=10.0))
