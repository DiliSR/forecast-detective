"""Tests for the tools and the agent loop with a scripted fake LLM (no network, no real explanations)."""

import json
from types import SimpleNamespace

import pytest

from forecast_explainer import agent, agent_tools, evidence as ev


# ----------------------------------------------------------------------------- a scripted tool-calling LLM

def tool_turn(*calls, content=None):
    """One scripted assistant turn that calls tools: calls are (name, args dict)."""
    tcs = [SimpleNamespace(id=f"call{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a) if isinstance(a, dict) else a))
           for i, (n, a) in enumerate(calls)]
    return SimpleNamespace(content=content, tool_calls=tcs)


def text_turn(text):
    return SimpleNamespace(content=text, tool_calls=None)


class FakeLLM:
    """Plays back scripted turns in order and records every request it received."""
    def __init__(self, turns):
        self.turns, self.requests = list(turns), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        msg = self.turns[min(len(self.requests) - 1, len(self.turns) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], model="fake", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


@pytest.fixture
def stub_narrate(monkeypatch):
    """Replace the writer (which would need explanations, an index and an LLM) by a stub returning a plausible result."""
    calls = []

    def fake(ts, models="both", question=None, model=None, use_rag=True, llm_client=None, **kw):
        calls.append({"ts": str(ts), "models": ev.normalize_models(models), "question": question, "use_rag": use_rag})
        return {"timestamp": str(ts), "models": list(ev.normalize_models(models)), "model": "fake", "text": "explanation [E:forecast]",
                "checks": {"passed": True, "n_citations": 1, "unsupported_numbers": [], "unknown_citations": [], "flags_not_cited": False},
                "sources": []}
    monkeypatch.setattr(agent_tools.narrate, "narrate", fake)
    return calls


# ----------------------------------------------------------------------------- tools

def test_get_forecasts_returns_24_hours_and_the_worst_hour():
    out = agent_tools.ToolBox().get_forecasts("2014-12-24")
    assert len(out["hours"]) == 24 and out["weekday"] == "Wednesday"
    assert set(out["largest_error_hour"]) == {"lightgbm", "lstm"}


def test_get_forecasts_outside_the_range_explains_the_range():
    out = agent_tools.ToolBox().get_forecasts("2013-12-24")
    assert "error" in out and "2014-07-01" in out["error"]


def test_explain_hour_validates_hour_and_model(stub_narrate):
    box = agent_tools.ToolBox()
    assert "error" in box.explain_hour("2013-12-24 18:00")
    assert "error" in box.call("explain_hour", {"timestamp": "2014-12-24 18:00", "models": "xgboost"})
    assert stub_narrate == []                                         # nothing was run for invalid requests


def test_explain_hour_limit_per_request(stub_narrate):
    box = agent_tools.ToolBox()
    for h in range(agent_tools.MAX_EXPLANATIONS_PER_REQUEST):
        assert box.explain_hour(f"2014-12-24 {h + 10}:00")["status"] == "done"
    assert "At most" in box.explain_hour("2014-12-24 20:00")["error"]


def test_unknown_tool_and_bad_arguments_become_errors():
    box = agent_tools.ToolBox()
    assert "unknown tool" in box.call("delete_everything", {})["error"]
    assert "error" in box.call("get_forecasts", {"wrong": 1})


# ----------------------------------------------------------------------------- the loop

def test_simple_request_runs_explain_hour_and_returns_the_narrative(stub_narrate):
    fake = FakeLLM([tool_turn(("explain_hour", {"timestamp": "2014-12-24 18:00", "models": "lstm"})),
                    text_turn("Explained 24 Dec 18:00 for the LSTM only.")])
    out = agent.run_agent("explain the LSTM on christmas eve at 6pm", llm_client=fake)
    assert out["stopped"] == "done" and out["message"].startswith("Explained")
    assert [n["models"] for n in out["narratives"]] == [["lstm"]]
    assert out["trace"][0]["tool"] == "explain_hour" and out["usage"]["prompt_tokens"] == 20
    assert "[explained: 2014-12-24 18:00 (lstm)]" in out["history_text"]


def test_vague_request_looks_at_the_day_first(stub_narrate):
    fake = FakeLLM([tool_turn(("get_forecasts", {"date": "2014-12-24"})),
                    tool_turn(("explain_hour", {"timestamp": "2014-12-24 20:00", "question": "why so high?"})),
                    text_turn("I chose 20:00, the largest error.")])
    out = agent.run_agent("why was Christmas Eve so wrong?", llm_client=fake)
    assert [t["tool"] for t in out["trace"]] == ["get_forecasts", "explain_hour"]
    assert stub_narrate[0]["question"] == "why so high?" and stub_narrate[0]["models"] == ("lightgbm", "lstm")
    tool_reply = [m for m in fake.requests[1]["messages"] if m["role"] == "tool"][0]
    assert "largest_error_hour" in tool_reply["content"]              # the LLM was shown the day table


def test_ambiguous_request_ends_with_a_question(stub_narrate):
    fake = FakeLLM([tool_turn(("ask_user", {"question": "Christmas Eve (24th) or Christmas Day (25th)?"}))])
    out = agent.run_agent("explain the forecast at christmas", llm_client=fake)
    assert out["stopped"] == "clarify" and "24th" in out["clarification"] and out["narratives"] == []
    assert len(fake.requests) == 1                                    # stopped without another LLM turn


def test_tool_error_is_given_back_to_the_llm_which_can_recover(stub_narrate):
    fake = FakeLLM([tool_turn(("explain_hour", {"timestamp": "2013-12-24 18:00"})),
                    tool_turn(("explain_hour", {"timestamp": "2014-12-24 18:00"})),
                    text_turn("Used 2014, the only year in range.")])
    out = agent.run_agent("explain christmas eve", llm_client=fake)
    assert "error" in out["trace"][0]["result"] and out["trace"][1]["result"]["status"] == "done"
    assert len(out["narratives"]) == 1


def test_malformed_tool_arguments_do_not_crash(stub_narrate):
    fake = FakeLLM([tool_turn(("explain_hour", "{not json")), text_turn("Sorry, I could not do that.")])
    out = agent.run_agent("explain something", llm_client=fake)
    assert "not valid JSON" in out["trace"][0]["result"]["error"] and out["stopped"] == "done"


def test_step_limit_stops_a_looping_agent(stub_narrate):
    fake = FakeLLM([tool_turn(("get_forecasts", {"date": "2014-12-24"}))])   # never stops calling tools
    out = agent.run_agent("loop", llm_client=fake, max_steps=3)
    assert out["stopped"] == "max_steps" and len(fake.requests) == 3


def test_history_is_sent_so_follow_ups_work(stub_narrate):
    fake = FakeLLM([text_turn("ok")])
    history = [{"role": "user", "content": "explain 24 dec 18:00"}, {"role": "assistant", "content": "Done [explained: 2014-12-24 18:00 (lightgbm+lstm)]"}]
    agent.run_agent("now only the LSTM", history=history, llm_client=fake)
    sent = [m["content"] for m in fake.requests[0]["messages"]]
    assert "explain 24 dec 18:00" in sent and "now only the LSTM" in sent   # earlier turns come before the new message
    assert sent.index("explain 24 dec 18:00") < sent.index("now only the LSTM")


def test_system_prompt_contains_the_real_date_range():
    prompt = agent.agent_system_prompt(agent_tools.ToolBox())
    assert "2014-07-01 00:00" in prompt and "2014-12-31 23:00" in prompt


def test_identical_tool_calls_in_one_run_are_answered_once(stub_narrate):
    fake = FakeLLM([tool_turn(("get_forecasts", {"date": "2014-12-24"}), ("get_forecasts", {"date": "2014-12-24"})),
                    text_turn("ok")])
    out = agent.run_agent("x", llm_client=fake)
    assert len(out["trace"]) == 2 and out["trace"][0]["result"] == out["trace"][1]["result"]


def test_the_same_hour_cannot_be_explained_twice_with_different_models(stub_narrate):
    fake = FakeLLM([tool_turn(("explain_hour", {"timestamp": "2014-10-16 12:00", "models": "both"}),
                              ("explain_hour", {"timestamp": "2014-10-16 12:00", "models": "lstm"})),
                    text_turn("done")])
    out = agent.run_agent("explain 16 october at noon", llm_client=fake)
    assert len(out["narratives"]) == 1 and len(stub_narrate) == 1
    assert "already explained" in out["trace"][1]["result"]["error"]


def test_on_step_callback_reports_each_tool_call(stub_narrate):
    seen = []
    fake = FakeLLM([tool_turn(("get_forecasts", {"date": "2014-12-24"})), text_turn("ok")])
    agent.run_agent("x", llm_client=fake, on_step=seen.append)
    assert [s["tool"] for s in seen] == ["get_forecasts"]


# ----------------------------------------------------------------------------- the planner tool

@pytest.fixture
def stub_planner(monkeypatch, make_recommendation):
    """Replace planner.recommend (which would call the writer LLM) by the real analysis with a canned text."""
    calls = []

    def fake(start=None, end=None, focus=None, question=None, **kw):
        calls.append({"start": start, "end": end, "focus": focus, "question": question})
        return make_recommendation(start, end, focus, question)
    monkeypatch.setattr(agent_tools.planner, "recommend", fake)
    return calls


def test_a_model_question_calls_the_planner_and_returns_the_recommendation(stub_planner):
    fake = FakeLLM([tool_turn(("recommend_model", {"focus": "evening 18-23", "question": "is the LSTM better in the evening?"})),
                    text_turn("Compared the evening hours.")])
    out = agent.run_agent("is the LSTM better in the evening?", llm_client=fake)
    assert stub_planner == [{"start": None, "end": None, "focus": "evening 18-23", "question": "is the LSTM better in the evening?"}]
    assert out["narratives"] == [] and len(out["recommendations"]) == 1
    assert out["recommendations"][0]["scope"]["focus"] == "evening 18-23"
    assert "[compared models: evening 18-23 hours" in out["history_text"]
    reply = json.loads([m for m in fake.requests[1]["messages"] if m["role"] == "tool"][0]["content"])
    assert reply["status"] == "done" and "Do not repeat" in reply["note"] and "headline" not in reply   # the agent never sees the text


def test_the_agent_can_both_compare_and_explain_in_one_request(stub_narrate, stub_planner):
    fake = FakeLLM([tool_turn(("recommend_model", {}), ("explain_hour", {"timestamp": "2014-12-25 09:00"})), text_turn("Did both.")])
    out = agent.run_agent("which model is better overall, and why was 25 Dec 09:00 off?", llm_client=fake)
    assert len(out["recommendations"]) == 1 and len(out["narratives"]) == 1


def test_planner_errors_are_given_back_to_the_agent(stub_planner):
    box = agent_tools.ToolBox()
    assert "test period" in box.call("recommend_model", {"start_date": "2013-01-01"})["error"]
    assert "unknown focus" in box.call("recommend_model", {"focus": "midnight"})["error"]
    assert stub_planner == [{"start": "2013-01-01", "end": None, "focus": None, "question": None},
                            {"start": None, "end": None, "focus": "midnight", "question": None}]
    assert box.recommendations == []


def test_recommendations_per_request_are_limited(stub_planner):
    box = agent_tools.ToolBox()
    assert agent_tools.MAX_RECOMMENDATIONS_PER_REQUEST == 2
    assert box.recommend_model(focus="weekend")["status"] == "done"
    assert box.recommend_model(focus="weekday")["status"] == "done"
    assert "At most" in box.recommend_model(focus="night 00-05")["error"]


def test_the_tool_schema_offers_exactly_the_planner_focus_choices():
    schema = next(t for t in agent_tools.TOOLS if t["function"]["name"] == "recommend_model")
    assert schema["function"]["parameters"]["properties"]["focus"]["enum"] == agent_tools.planner.FOCUS_CHOICES


def test_system_prompt_routes_model_questions_to_the_planner():
    prompt = agent.agent_system_prompt(agent_tools.ToolBox())
    assert "recommend_model" in prompt and '"evening 18-23"' in prompt and "never judge it yourself" in prompt
