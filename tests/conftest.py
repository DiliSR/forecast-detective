"""Fixtures shared by several test files."""

import pytest

from forecast_explainer import planner


@pytest.fixture
def make_recommendation():
    """A factory for a planner.recommend()-shaped result: the real deterministic analysis plus a canned text.

    Building it on the real analysis (planner.analyse, about 0.2 s, no LLM) keeps the decision, slices and evidence
    blocks realistic, so the app draws real charts. Dates outside the test period raise ValueError like the real planner.
    """
    def build(start=None, end=None, focus=None, question=None, passed=True, **ignored):
        analysis = planner.analyse(start, end, focus, with_explanations=False)
        return {**analysis, "kind": "recommendation", "question": question,
                "text": "**Why**\n- LightGBM has the lower RMSE [E:overall].", "n_attempts": 1, "model": "fake-llm", "use_rag": True,
                "checks": {"passed": passed, "n_citations": 1, "unknown_citations": [], "uncited_numbers": [], "flags_not_cited": False,
                           "unsupported_numbers": [] if passed else [{"number": "9.9", "sentence": "x"}], "causal_without_source": [], "words": 12},
                "evidence": planner.evidence_blocks(analysis),
                "sources": [{"sid": "S1", "citation": "Note: model_cards", "page": 0, "section": "LSTM", "distance": 0.3, "text": "The LSTM sees 168 hours."}],
                "queries": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
    return build
