"""Run the Streamlit page in-process with AppTest and stubbed writer/agent (no network, no LLM).

AppTest cannot click a point on a chart (Streamlit documents this), so chart clicks are covered by the plain tests of
`view.hour_from_selection` in test_view.py; everything else on the page is exercised here.
"""

import json
import os
from datetime import date

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from forecast_explainer import PROJECT_ROOT, agent, narrate, planner, store

TS = "2014-12-24 18:00"
APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")


def fake_narrative(ts=TS, models=("lightgbm", "lstm"), passed=True, causal=()):
    """A stand-in for narrate.narrate()'s result, built on the real cached explanation so the charts have data."""
    store.get_explanation(ts, verbose=False)
    return {
        "timestamp": str(pd.Timestamp(ts)), "models": list(models), "model": "fake-llm", "use_rag": True, "n_attempts": 1,
        "text": "**Summary** Both over-forecast [E:forecast] [E:flags].",
        "checks": {"passed": passed, "n_citations": 2, "unknown_citations": [], "uncited_numbers": [], "flags_not_cited": False,
                   "unsupported_numbers": [] if passed else [{"number": "31,000", "sentence": "x"}], "causal_without_source": list(causal)},
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "evidence": {"E:forecast": "Target hour ...", "E:flags": "flags ..."},
        "sources": [{"sid": "S1", "citation": "Note: calendar_effects", "page": 0, "section": "Christmas Eve", "distance": 0.3, "text": "Load drops."}],
    }


def fake_agent(narratives=(), stopped="done", message="ok", clarification=None, recommendations=()):
    return {"message": message, "stopped": stopped, "clarification": clarification, "narratives": list(narratives),
            "recommendations": list(recommendations),
            "trace": [{"tool": "explain_hour", "arguments": {"timestamp": TS}, "result": {}, "seconds": 1.0}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "history_text": f"{message} [explained]"}


def new_app(timeout=60):
    return AppTest.from_file(APP, default_timeout=timeout).run()


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    if not store.explanation_path(TS).exists():
        pytest.skip("explanation for the test hour not cached")
    monkeypatch.setenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY") or "test-key")


def chart_titles(at):
    """Titles of the drivers charts on the page (the forecast chart has none)."""
    specs = [json.loads(c.proto.spec) for c in at.get("vega_lite_chart")]
    return [sp["title"] for sp in specs if isinstance(sp.get("title"), str)]


def texts(elements):
    return " ".join(str(getattr(e, "value", "")) for e in elements)


# ----------------------------------------------------------------------------- the first view

def test_first_view_shows_the_forecast_chart_the_misses_and_the_selected_hour():
    at = new_app()
    assert not at.exception and at.title[0].value == "Forecast Detective"
    assert len(at.get("vega_lite_chart")) == 1                                   # the forecast time series comes first
    assert [b.label for b in at.button if b.key and b.key.startswith("spot")][:2] == ["31 Dec 20:00 · +80,593 kWh", "24 Dec 20:00 · +79,610 kWh"]
    assert [(m.label, m.value) for m in at.metric] == [("Actual", "211,763 kWh"), ("LightGBM", "242,680 kWh"), ("LSTM", "243,750 kWh")]
    assert at.metric[1].delta == "+14.6% vs actual"
    assert at.session_state["pick_hour"] == 18 and at.session_state["pick_date"] == date(2014, 12, 24)
    assert "Choose an hour and press Explain" in texts(at.caption)               # nothing is explained until asked


def test_the_chart_spec_has_the_click_selection_and_alt_text():
    spec = new_app().get("vega_lite_chart")[0].proto.spec
    assert '"pick"' in spec and '"idx"' in spec                                 # click selection on the integer hour index
    assert "nearest" in spec


# ----------------------------------------------------------------------------- choosing an hour

def test_a_largest_miss_button_moves_the_pickers_and_the_numbers():
    at = new_app()
    at.button(key="spot4257").click().run()                                      # 25 Dec 09:00
    assert not at.exception
    assert at.session_state["pick_date"] == date(2014, 12, 25) and at.session_state["pick_hour"] == 9
    assert at.metric[0].value == "91,488 kWh"


def test_changing_the_day_or_hour_by_hand_updates_the_cards():
    at = new_app()
    at.date_input(key="pick_date").set_value(date(2014, 10, 16)).run()
    at.select_slider(key="pick_hour").set_value(12).run()
    assert not at.exception and at.metric[0].value == "282,459 kWh"


def test_changing_the_hour_hides_an_explanation_that_no_longer_matches(monkeypatch):
    monkeypatch.setattr(narrate, "narrate", lambda *a, **k: fake_narrative())
    at = explain(new_app())
    assert at.session_state["results"]
    at.select_slider(key="pick_hour").set_value(10).run()
    assert at.session_state["results"] == []


# ----------------------------------------------------------------------------- explain this hour

def explain(at):
    next(b for b in at.button if b.label == "Explain this hour").click().run()
    return at


def test_explain_uses_the_chosen_hour_and_model_and_shows_a_short_answer(monkeypatch):
    calls = []
    monkeypatch.setattr(narrate, "narrate", lambda ts, **kw: calls.append((ts, kw)) or fake_narrative(models=("lstm",)))
    at = new_app()
    at.session_state["model_choice"] = "LSTM"
    explain(at.run())
    assert not at.exception
    assert calls[0][0] == pd.Timestamp(TS) and calls[0][1]["models"] == "lstm" and calls[0][1]["use_rag"] is True
    assert "Both over-forecast" in texts(at.markdown)
    assert chart_titles(at) == ["LSTM · Integrated Gradients", "LSTM · TimeSHAP"]  # only the explained model's charts


def test_both_models_get_both_sets_of_drivers_charts(monkeypatch):
    monkeypatch.setattr(narrate, "narrate", lambda *a, **k: fake_narrative())
    at = explain(new_app())
    assert chart_titles(at) == ["LightGBM · SHAP", "LSTM · Integrated Gradients", "LightGBM · LIME", "LSTM · TimeSHAP"]


def test_a_failed_check_is_shown_not_hidden(monkeypatch):
    monkeypatch.setattr(narrate, "narrate", lambda *a, **k: fake_narrative(passed=False, causal=["It was due to X [E:flags]."]))
    at = explain(new_app())
    assert not at.exception
    badges = texts(at.markdown)
    assert "Checks failed" in badges and "Cause without source" in badges
    assert "could not be fully verified" in texts(at.warning) and "31,000" in texts(at.warning)


def test_an_api_error_does_not_crash_the_page(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("API is down")
    monkeypatch.setattr(narrate, "narrate", boom)
    at = explain(new_app())
    assert not at.exception and "API is down" in texts(at.error)


# ----------------------------------------------------------------------------- ask in words

def test_a_question_goes_to_the_agent_and_the_page_follows_the_hour_it_chose(monkeypatch):
    seen = []
    result = fake_agent([fake_narrative("2014-12-25 18:00", models=("lstm",))])
    monkeypatch.setattr(agent, "run_agent", lambda prompt, history, **kw: seen.append((prompt, list(history), kw)) or result)
    at = new_app()
    at.session_state["queued_ask"] = "explain the LSTM on christmas day at 6pm"
    at.run()
    assert not at.exception and seen[0][0] == "explain the LSTM on christmas day at 6pm" and seen[0][1] == []
    assert (at.session_state["pick_date"], at.session_state["pick_hour"], at.session_state["model_choice"]) == (date(2014, 12, 25), 18, "LSTM")
    assert at.session_state["results"] and at.session_state["trace"]
    assert at.metric[0].value == "124,156 kWh"                                  # cards now show the hour the agent chose


def test_follow_up_questions_carry_the_conversation(monkeypatch):
    seen = []
    monkeypatch.setattr(agent, "run_agent", lambda prompt, history, **kw: seen.append(list(history)) or fake_agent([fake_narrative()]))
    at = new_app()
    at.session_state["queued_ask"] = "first"
    at.run()
    at.session_state["queued_ask"] = "second"
    at.run()
    assert [m["role"] for m in seen[1]] == ["user", "assistant"]


def test_an_ambiguous_question_shows_the_agents_question(monkeypatch):
    monkeypatch.setattr(agent, "run_agent", lambda *a, **k: fake_agent(stopped="clarify", message="Which day?", clarification="The 24th or the 25th?"))
    at = new_app()
    at.session_state["queued_ask"] = "explain christmas"
    at.run()
    assert not at.exception and "The 24th or the 25th?" in texts(at.info) and at.session_state["results"] == []


def test_a_request_the_agent_cannot_do_shows_its_message(monkeypatch):
    monkeypatch.setattr(agent, "run_agent", lambda *a, **k: fake_agent(message="I can only explain specific hours."))
    at = new_app()
    at.session_state["queued_ask"] = "which model is best?"
    at.run()
    assert "I can only explain specific hours." in texts(at.info)


def test_agent_errors_do_not_crash_the_page(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no network")
    monkeypatch.setattr(agent, "run_agent", boom)
    at = new_app()
    at.session_state["queued_ask"] = "anything"
    at.run()
    assert not at.exception and "no network" in texts(at.error)


# ----------------------------------------------------------------------------- which model?

def test_compare_models_here_sends_the_chart_window_to_the_planner(monkeypatch, make_recommendation):
    seen = []
    monkeypatch.setattr(planner, "recommend", lambda start, end, **kw: seen.append((start, end)) or make_recommendation(start, end))
    at = new_app()                                                                 # default window: Christmas week
    next(b for b in at.button if b.label == "Compare models here").click().run()
    assert not at.exception and seen == [("2014-12-20", "2014-12-31")]            # clamped to the end of the test period
    assert at.session_state["recs"] and at.session_state["results"] == []
    assert "Which model?" in texts(at.markdown) and "Checks passed" in texts(at.markdown)
    assert "Use " in texts(at.success) or "No clear" in texts(at.info) or "Too few" in texts(at.warning)   # the code-made headline


def test_the_whole_period_window_compares_the_whole_test_period(monkeypatch, make_recommendation):
    seen = []
    monkeypatch.setattr(planner, "recommend", lambda start, end, **kw: seen.append((start, end)) or make_recommendation(start, end))
    at = new_app()
    at.session_state["view"] = "Whole period"
    at.run()
    next(b for b in at.button if b.label == "Compare models here").click().run()
    assert seen == [(None, None)]
    assert "No clear winner" in texts(at.info)                                    # under RMSE the overall gap is not clear: a blue box


def test_a_custom_range_is_compared_exactly_as_chosen(monkeypatch, make_recommendation):
    seen = []
    monkeypatch.setattr(planner, "recommend", lambda start, end, **kw: seen.append((start, end)) or make_recommendation(start, end))
    at = new_app()
    at.session_state["view"] = "Custom range"
    at.run()
    assert not at.exception and [d.label for d in at.date_input][0] == "Range"           # the range picker appears
    at.session_state["custom_range"] = (date(2014, 10, 20), date(2014, 11, 2))
    at.run()
    next(b for b in at.button if b.label == "Compare models here").click().run()
    assert not at.exception and seen == [("2014-10-20", "2014-11-02")]


def test_a_one_day_window_shows_a_warning_not_a_recommendation(monkeypatch, make_recommendation):
    monkeypatch.setattr(planner, "recommend", lambda start, end, **kw: make_recommendation("2014-12-25", "2014-12-25"))
    at = new_app()
    next(b for b in at.button if b.label == "Compare models here").click().run()
    assert "Too few days" in texts(at.warning) and "Use " not in texts(at.success)


def test_a_failed_check_on_a_recommendation_is_shown(monkeypatch, make_recommendation):
    monkeypatch.setattr(planner, "recommend", lambda start, end, **kw: make_recommendation(start, end, passed=False))
    at = new_app()
    next(b for b in at.button if b.label == "Compare models here").click().run()
    assert "Checks failed" in texts(at.markdown)


def test_a_planner_error_does_not_crash_the_page(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("planner is down")
    monkeypatch.setattr(planner, "recommend", boom)
    at = new_app()
    next(b for b in at.button if b.label == "Compare models here").click().run()
    assert not at.exception and "planner is down" in texts(at.error)


def test_a_model_question_to_the_agent_shows_the_recommendation(monkeypatch, make_recommendation):
    result = fake_agent(recommendations=[make_recommendation(focus="evening 18-23")], message="Compared the evenings.")
    monkeypatch.setattr(agent, "run_agent", lambda *a, **k: result)
    at = new_app()
    at.session_state["queued_ask"] = "which model is better in the evening?"
    at.run()
    assert not at.exception and at.session_state["recs"] and at.session_state["results"] == []
    assert "evening 18-23" in texts(at.success) + texts(at.info) + texts(at.warning)


def test_changing_the_hour_keeps_a_model_recommendation(monkeypatch, make_recommendation):
    monkeypatch.setattr(planner, "recommend", lambda start, end, **kw: make_recommendation(start, end))
    at = new_app()
    next(b for b in at.button if b.label == "Compare models here").click().run()
    at.select_slider(key="pick_hour").set_value(10).run()
    assert at.session_state["recs"]                                                 # it is about the window, not the hour


def test_compare_is_disabled_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert next(b for b in new_app().button if b.label == "Compare models here").disabled


# ----------------------------------------------------------------------------- no key

def test_without_an_openai_key_the_forecast_still_shows_and_explaining_is_off(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = new_app()
    assert not at.exception and len(at.get("vega_lite_chart")) == 1
    assert next(b for b in at.button if b.label == "Explain this hour").disabled
    assert "OPENAI_API_KEY" in texts(at.sidebar.warning)
