"""Explaining one model: the evidence, flags, retrieval queries and prompt must not mention the other model."""

import pytest

from forecast_explainer import evidence as ev, narrate as nar, store
from forecast_explainer.rag.queries import queries_from_evidence

TS = "2014-12-24 18:00"


@pytest.fixture(scope="module")
def explanation():
    if not store.explanation_path(TS).exists():
        pytest.skip("explanation for the test hour not cached")
    return store.get_explanation(TS, verbose=False)


def test_normalize_models():
    assert ev.normalize_models(None) == ev.normalize_models("both") == ("lightgbm", "lstm")
    assert ev.normalize_models("lstm") == ("lstm",)
    assert ev.normalize_models(["lstm", "lightgbm"]) == ("lightgbm", "lstm")
    with pytest.raises(ValueError):
        ev.normalize_models("xgboost")


def test_lstm_only_evidence_has_no_lightgbm_blocks(explanation):
    blocks = ev.build_evidence(explanation, models="lstm")
    assert list(blocks) == ["E:forecast", "E:ig", "E:timeshap", "E:flags", "E:accuracy"]
    text = " ".join(blocks.values())
    assert "LightGBM" not in text and "SHAP for LightGBM" not in text and "LIME" not in text


def test_lightgbm_only_evidence_has_no_lstm_blocks(explanation):
    blocks = ev.build_evidence(explanation, models="lightgbm")
    assert list(blocks) == ["E:forecast", "E:shap", "E:lime", "E:flags", "E:accuracy"]
    assert "LSTM" not in " ".join(blocks.values()) and "differ by" not in blocks["E:forecast"]


def test_single_model_flags_only_concern_that_model(explanation):
    flags = ev.restrict_to_models(explanation, "lstm")["disagreements"]
    assert {f["type"] for f in flags} <= ev.FLAGS_OF["lstm"]
    assert {f["type"] for f in ev.restrict_to_models(explanation, "lightgbm")["disagreements"]} <= ev.FLAGS_OF["lightgbm"]


def test_queries_follow_the_chosen_model(explanation):
    lstm = " ".join(q["query"] for q in queries_from_evidence(explanation, "lstm"))
    assert "LSTM" in lstm and "SHAP values add up" not in lstm and "LIME" not in lstm
    both = " ".join(q["query"] for q in queries_from_evidence(explanation, "both"))
    assert "SHAP values add up" in both and "LIME" in both


def test_system_prompt_for_one_model_does_not_introduce_the_other():
    one = nar.system_prompt("lstm")
    assert "LSTM" in one and "LightGBM" not in one and "Do not discuss any other model" in one
    both = nar.system_prompt("both")
    assert "LightGBM" in both and "LSTM" in both and "for each model" in both


def test_cache_path_differs_by_model_choice():
    paths = {m: nar.narrative_path(TS, "total", "gpt", True, None, m) for m in ("both", "lstm", "lightgbm")}
    assert len(set(paths.values())) == 3 and "_both_" in paths["both"].name
