"""Tests for the planner: the statistics on synthetic data (where the right answer is known by construction), the scope
validation and the decision on the real test period, and the writer with a scripted fake LLM (no network)."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from forecast_explainer import checks, planner as pl


# ----------------------------------------------------------------------------- synthetic frames

def synthetic(extra, hours=24, base=1000.0):
    """A frame shaped like planner.hours_in_scope output: one block of `hours` rows per day.

    extra: for each day, how much BIGGER the LSTM's miss is than LightGBM's, as a fraction (0.5 = 50% bigger, negative =
    the LSTM missed by less). LightGBM misses by `base` kWh every hour, so the truth of "who is better" is known.
    """
    rows = []
    for d, e in enumerate(extra):
        lstm = base * (1 + e)
        rows.append({"day": pd.Timestamp("2014-08-01") + pd.Timedelta(days=d), "n": hours,
                     "ae_lightgbm": base, "ae_lstm": lstm, "se_lightgbm": base ** 2, "se_lstm": lstm ** 2,
                     "err_lightgbm": 2.0, "err_lstm": 2.0})
    return pd.DataFrame(rows).loc[lambda f: f.index.repeat(f["n"])].drop(columns="n").reset_index(drop=True)


# ----------------------------------------------------------------------------- compare_models: the statistics

def test_a_model_that_is_better_every_day_is_clearly_better():
    r = pl.compare_models(synthetic([0.5, 0.6, 0.4, 0.5] * 10))
    assert r["confidence"] == "clear" and r["winner"] == "lightgbm" and r["gain_ci"][0] > 0
    assert r["n_days"] == 40 and r["rmse"]["lstm"] > r["rmse"]["lightgbm"]


def test_the_winner_can_be_the_lstm():
    r = pl.compare_models(synthetic([-0.5, -0.6, -0.4, -0.5] * 10))
    assert r["winner"] == "lstm" and r["leaning"] == "lstm"


def test_a_gap_that_flips_sign_between_days_is_not_clear():
    # twenty days where the LSTM misses 50% more, twenty where LightGBM does: the totals are equal by construction
    lstm_worse = synthetic([0.5] * 20)
    swap = {"ae_lightgbm": "ae_lstm", "ae_lstm": "ae_lightgbm", "se_lightgbm": "se_lstm", "se_lstm": "se_lightgbm"}
    lightgbm_worse = lstm_worse.rename(columns=swap).assign(day=lambda f: f["day"] + pd.Timedelta(days=20))
    r = pl.compare_models(pd.concat([lstm_worse, lightgbm_worse], ignore_index=True))
    assert r["confidence"] == "no_clear_difference" and r["winner"] is None and r["gain_ci"][0] < 0 < r["gain_ci"][1]


def test_a_gap_below_the_minimum_effect_is_not_worth_acting_on():
    r = pl.compare_models(synthetic([pl.MIN_EFFECT / 200] * 30))   # LSTM RMSE ~1.5% higher every day: the interval excludes 0 but the gap is tiny
    assert r["gain_ci"][0] > 0 and r["confidence"] == "no_clear_difference"


def test_fewer_than_min_days_is_never_clear_however_big_the_gap():
    r = pl.compare_models(synthetic([9.0] * (pl.MIN_DAYS - 1)))
    assert r["confidence"] == "too_few_days" and r["gain_ci"] is None and r["winner"] is None
    assert r["leaning"] == "lightgbm"                           # the lean is still reported, labelled as unreliable


def test_hours_of_one_day_do_not_count_as_independent_evidence():
    # 4 days of 24 hours = 96 hours would look huge if hours were independent; days are what count.
    assert pl.compare_models(synthetic([5.0] * 4))["confidence"] == "too_few_days"


def test_the_interval_is_repeatable():
    frame = synthetic(list(np.random.default_rng(1).normal(0.05, 0.1, 40)))
    assert pl.compare_models(frame)["gain_ci"] == pl.compare_models(frame)["gain_ci"]


# ----------------------------------------------------------------------------- the policy

def row(label, leaning, confidence, n_days=30):
    other = "lstm" if leaning == "lightgbm" else "lightgbm"
    return {"by": "day_type", "label": label, "leaning": leaning, "confidence": confidence,
            "winner": leaning if confidence == "clear" else None, "n_days": n_days, "n_hours": n_days * 24,
            "rmse": {leaning: 2000.0, other: 3000.0}, "gain_pct": 33.3}


def test_policy_separates_reliable_switches_from_unreliable_ones():
    overall = {**row("all", "lightgbm", "clear", 100), "n_days": 100}
    pol = pl.plan_policy(overall, [row("weekend", "lstm", "clear"), row("major holiday", "lstm", "too_few_days", 1),
                                   row("weekday", "lightgbm", "clear"), row("evening 18-23", "lstm", "no_clear_difference")])
    assert pol["default"] == "lightgbm"
    assert [s["label"] for s in pol["switch_to"]] == ["weekend"]
    assert [s["label"] for s in pol["possible_switch"]] == ["major holiday"]     # no clear difference is neither


def test_policy_is_empty_when_the_scope_is_too_short_to_have_a_default():
    overall = {**row("all", "lstm", "too_few_days", 2), "n_days": 2}
    pol = pl.plan_policy(overall, [row("night 00-05", "lightgbm", "too_few_days", 2)])
    assert pol["switch_to"] == [] and pol["possible_switch"] == []


# ----------------------------------------------------------------------------- scope (real test period)

def test_the_whole_test_period_is_the_default_scope():
    f = pl.hours_in_scope()
    assert len(f) == 4416 and f["day"].nunique() == 184


def test_end_day_is_inclusive_and_focus_filters():
    assert len(pl.hours_in_scope("2014-12-24", "2014-12-25")) == 48
    evening = pl.hours_in_scope("2014-12-01", "2014-12-07", focus="evening 18-23")
    assert set(evening.index.hour) == {18, 19, 20, 21, 22, 23}
    assert set(pl.hours_in_scope(focus="major holiday")["day"].dt.strftime("%m-%d")) == {"12-25"}


@pytest.mark.parametrize("kwargs, message", [
    ({"start": "2013-12-24"}, "test period"), ({"end": "2015-02-01"}, "test period"),
    ({"start": "2014-12-02", "end": "2014-12-01"}, "after end"), ({"focus": "night"}, "unknown focus"),
    ({"start": "2014-07-04", "end": "2014-07-04", "focus": "major holiday"}, "No hours match")])
def test_bad_scopes_raise_readable_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pl.hours_in_scope(**kwargs)


# ----------------------------------------------------------------------------- the decision on the real data

def test_overall_gap_is_within_the_noise_night_is_clear_and_christmas_day_is_flagged_as_unreliable():
    a = pl.analyse(with_explanations=False)
    assert a["decision"]["model"] is None and a["decision"]["leaning"] == "lightgbm"     # under RMSE the overall gap is within the noise
    night = next(r for r in a["slices"] if r["label"] == "night 00-05")
    assert night["winner"] == "lightgbm" and night["confidence"] == "clear"
    holiday = next(r for r in a["slices"] if r["label"] == "major holiday")
    assert holiday["n_days"] == 1 and holiday["confidence"] == "too_few_days" and holiday["leaning"] == "lstm"
    assert any("major holiday" in s and "not reliable" in s for s in a["decision"]["switches"])
    assert not a["policy"]["switch_to"] or all(s["label"] != "major holiday" for s in a["policy"]["switch_to"])


def test_a_single_day_never_gets_a_recommendation():
    a = pl.analyse("2014-12-25", "2014-12-25", with_explanations=False)
    d = a["decision"]
    assert d["model"] is None and d["confidence"] == "too_few_days" and "1 day)" in d["headline"] and d["switches"] == []


def test_the_analysis_is_json_serialisable():
    json.dumps(pl.analyse(with_explanations=False))


def test_worst_hours_without_explanations_has_no_drivers_and_with_them_the_top_two(monkeypatch):
    frame = pl.hours_in_scope("2014-12-24", "2014-12-25")
    assert all(w["drivers"] is None for w in pl.worst_hours(frame, with_explanations=False).values())
    records = [{"feature": "a", "attribution_kwh": 5.0}, {"feature": "b", "attribution_kwh": -9.0},
               {"feature": "c", "attribution_kwh": 1.0}, {"feature": "z", "attribution_kwh": 0.2}]
    monkeypatch.setattr(pl.store, "get_explanation",
                        lambda ts, verbose=False: {"lightgbm": {"shap": records}, "lstm": {"ig_by_feature": records}})
    drivers = pl.worst_hours(frame, with_explanations=True)["lightgbm"]["drivers"]
    assert [d["feature"] for d in drivers] == ["b", "a"]


def test_a_failing_explanation_does_not_stop_the_planner(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no model")
    monkeypatch.setattr(pl.store, "get_explanation", boom)
    w = pl.worst_hours(pl.hours_in_scope("2014-12-24", "2014-12-25"), with_explanations=True)
    assert w["lightgbm"]["drivers"] is None and "no model" in w["lightgbm"]["drivers_error"]


# ----------------------------------------------------------------------------- evidence blocks

def test_evidence_blocks_have_the_expected_tags_and_singular_days():
    blocks = pl.evidence_blocks(pl.analyse("2014-12-25", "2014-12-25", with_explanations=False))
    assert list(blocks) == ["E:scope", "E:overall", "E:slices", "E:policy", "E:worst", "E:limits"]
    assert "1 day " in blocks["E:scope"] and "1 days" not in " ".join(blocks.values())
    assert "No switching policy" in blocks["E:policy"]


def test_a_text_that_quotes_the_blocks_passes_the_checker_and_a_wrong_number_fails():
    a = pl.analyse(with_explanations=False)
    blocks = pl.evidence_blocks(a)
    good = "LightGBM has the lower RMSE, 10,296 kWh against 10,389 kWh for the LSTM [E:overall]."
    bad = "LightGBM has the lower RMSE, 9,000 kWh against 10,389 kWh for the LSTM [E:overall]."
    assert checks.check_text(good, blocks, [])["passed"]
    assert not checks.check_text(bad, blocks, [])["passed"]


# ----------------------------------------------------------------------------- the writer (fake LLM)

GOOD = ("**Why**\n- LightGBM has the lower RMSE, 10,296 kWh against 10,389 kWh for the LSTM [E:overall].\n"
        "**Caution**\n- Major holidays cover only 1 day [E:slices].")
WRONG_NUMBER = "**Why**\n- LightGBM has the lower RMSE, 9,000 kWh against 10,389 kWh for the LSTM [E:overall]."
TOO_LONG = GOOD + " " + "filler " * 120


class FakeClient:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], model="fake-model",
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "RECOMMENDATIONS_DIR", tmp_path)


def run(client, **kw):
    return pl.recommend(model="m", use_rag=False, with_explanations=False, llm_client=client, **kw)


def test_a_valid_text_passes_first_time_and_carries_the_decision(isolated):
    client = FakeClient([GOOD])
    out = run(client, question="which model?")
    assert out["checks"]["passed"] and out["n_attempts"] == 1 and out["source"] == "llm"
    assert out["decision"]["leaning"] == "lightgbm" and out["scope"]["n_days"] == 184
    user = client.calls[0]["messages"][1]["content"]
    assert "DECISION (made by code" in user and out["decision"]["headline"] in user and "which model?" in user


def test_a_wrong_number_is_retried_with_the_problem_fed_back(isolated):
    client = FakeClient([WRONG_NUMBER, GOOD])
    out = run(client)
    assert out["n_attempts"] == 2 and out["checks"]["passed"] and out["text"] == GOOD
    assert "9,000" in client.calls[1]["messages"][1]["content"]


def test_a_text_over_the_word_limit_is_sent_back_once(isolated):
    client = FakeClient([TOO_LONG, GOOD])
    out = run(client)
    assert out["n_attempts"] == 2 and out["checks"]["words"] <= pl.MAX_WORDS
    assert "the limit is 80" in client.calls[1]["messages"][1]["content"]


def test_judging_a_sample_as_substantial_or_reasonable_is_sent_back_once(isolated):
    judging = GOOD + " The 184 days are a substantial sample, not an insufficient one."
    client = FakeClient([judging, GOOD])
    out = run(client)
    assert out["n_attempts"] == 2 and out["checks"]["judging_words"] == [] and out["text"] == GOOD
    assert "substantial" in client.calls[1]["messages"][1]["content"]
    assert run(FakeClient([GOOD + " Insufficient data."]), refresh=True)["n_attempts"] == 1       # "insufficient" is allowed


def test_persistent_failure_is_reported_not_hidden(isolated):
    out = run(FakeClient([WRONG_NUMBER]))
    assert out["n_attempts"] == 2 and not out["checks"]["passed"]


def test_the_second_identical_question_comes_from_the_cache(isolated):
    client = FakeClient([GOOD])
    run(client)
    again = run(client)
    assert again["source"] == "cache" and len(client.calls) == 1
    assert run(client, question="a different question")["source"] == "llm"      # the wording is part of the cache key


def test_no_rag_sends_no_sources_and_rag_sends_some(isolated, monkeypatch):
    out = run(FakeClient([GOOD]))
    assert out["sources"] == [] and out["queries"] == []
    seen = []
    monkeypatch.setattr(pl, "search", lambda queries, **kw: seen.append(queries) or [
        {"id": "x", "text": "Note text", "citation": "Note: model_cards", "kind": "note", "page": 0, "section": "LSTM", "distance": 0.3}])
    out = pl.recommend(model="m", use_rag=True, with_explanations=False, llm_client=FakeClient([GOOD]), refresh=True)
    assert out["sources"][0]["sid"] == "S1" and any("LightGBM and LSTM accuracy" in q for q in seen[0])


def test_invalid_scope_raises_before_any_llm_call(isolated):
    client = FakeClient([GOOD])
    with pytest.raises(ValueError):
        run(client, focus="midnight")
    assert client.calls == []


def test_the_writer_prompt_forbids_causal_wording_and_does_not_ask_for_a_switch_section():
    assert "because, due to" in pl.SYSTEM and "Where to switch" not in pl.SYSTEM
