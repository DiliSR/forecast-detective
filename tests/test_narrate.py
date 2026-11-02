"""Tests for the explanation writer using a scripted fake LLM (no network). Needs the cached Christmas Eve
explanation (outputs/explanations) and the knowledge index; skipped if either is missing."""

from types import SimpleNamespace

import pytest

from forecast_explainer import llm, narrate as nar, store

TS = "2014-12-24 18:00"
GOOD = ("**Summary** Both models over-forecast the actual 211,763 kWh [E:forecast] [E:flags]. "
        "LightGBM leaned on `load_lag_168h` with +27,176 kWh [E:shap].")
BAD = "**Summary** Both models were off by 31,000 kWh [E:forecast] [E:flags]."


class FakeClient:
    """Mimics client.chat.completions.create and returns scripted answers in order."""
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], model="fake-model",
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    if not store.explanation_path(TS).exists():
        pytest.skip("explanation for the test hour not cached")
    monkeypatch.setattr(nar, "NARRATIVES_DIR", tmp_path)
    monkeypatch.setattr(nar, "narrative_path",
                        lambda ts, client, model, rag, q, *models: tmp_path / f"{client}_{model}_{rag}_{q}.json")
    try:
        from forecast_explainer.rag.ingest import get_collection
        get_collection()
    except RuntimeError:
        pytest.skip("knowledge index not built")


def test_valid_answer_passes_first_time():
    fake = FakeClient([GOOD])
    out = nar.narrate(TS, model="m", llm_client=fake)
    assert out["checks"]["passed"] and out["n_attempts"] == 1 and len(fake.calls) == 1
    assert out["source"] == "llm" and out["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}


def test_bad_first_answer_is_retried_with_the_problems_fed_back():
    fake = FakeClient([BAD, GOOD])
    out = nar.narrate(TS, model="m", llm_client=fake)
    assert out["n_attempts"] == 2 and out["checks"]["passed"] and out["text"] == GOOD
    retry_prompt = fake.calls[1]["messages"][1]["content"]
    assert "31,000" in retry_prompt and "not in the EVIDENCE" in retry_prompt


def test_persistent_failure_is_reported_not_hidden():
    out = nar.narrate(TS, model="m", llm_client=FakeClient([BAD]), max_retries=1)
    assert out["n_attempts"] == 2 and not out["checks"]["passed"]


def test_second_call_is_served_from_cache():
    fake = FakeClient([GOOD])
    nar.narrate(TS, model="m", llm_client=fake)
    again = nar.narrate(TS, model="m", llm_client=fake)
    assert again["source"] == "cache" and len(fake.calls) == 1


def test_no_rag_sends_no_sources():
    fake = FakeClient([GOOD])
    out = nar.narrate(TS, model="m", llm_client=fake, use_rag=False)
    assert out["sources"] == [] and "no sources provided" in fake.calls[0]["messages"][1]["content"]


def test_prompt_contains_evidence_ids_and_the_question():
    fake = FakeClient([GOOD])
    nar.narrate(TS, model="m", llm_client=fake, question="why so high?")
    user = fake.calls[0]["messages"][1]["content"]
    assert "QUESTION: why so high?" in user and "[E:shap]" in user and "[S1]" in user


def test_temperature_rejected_by_model_is_retried_without_it():
    class Picky(FakeClient):
        def create(self, **kwargs):
            if "temperature" in kwargs:
                raise RuntimeError("Unsupported value: 'temperature' does not support 0 with this model")
            return super().create(**kwargs)
    result = llm.complete("s", "u", model="m", client=Picky(["ok"]))
    assert result.text == "ok"


def test_retry_message_tells_the_writer_when_the_number_was_right_but_cited_wrongly():
    verdict = {"unknown_citations": [], "flags_not_cited": False, "unsupported_numbers": [
        {"number": "30,917", "sentence": "It was off by 30,917 kWh [S3].", "found_in": ["E:forecast"]},
        {"number": "99,999", "sentence": "It was 99,999 kWh.", "found_in": []}]}
    msg = nar.retry_message(verdict)
    assert "is correct but not in the block(s) cited" in msg and "[E:forecast]" in msg
    assert "99,999 is not in the EVIDENCE or SOURCES" in msg


def test_shap_evidence_separates_input_value_from_contribution():
    from forecast_explainer import evidence as ev
    if not store.explanation_path(TS).exists():
        pytest.skip("explanation not cached")
    block = ev.build_evidence(store.get_explanation(TS, verbose=False))["E:shap"]
    assert "input value" in block and "contribution" in block
