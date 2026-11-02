"""Tests for chunking and query building (no index or embedding model needed), plus one retrieval check."""

import pytest

from forecast_explainer.rag import ingest, queries
from forecast_explainer.rag.retrieve import format_context, label_hits


def test_short_text_is_one_chunk():
    assert ingest.chunk_words("one two three", size=10, overlap=2) == ["one two three"]


def test_empty_text_gives_no_chunks():
    assert ingest.chunk_words("   ", size=10, overlap=2) == []


def test_chunks_overlap_and_cover_everything():
    words = [f"w{i}" for i in range(50)]
    chunks = ingest.chunk_words(" ".join(words), size=20, overlap=5)
    assert all(len(c.split()) <= 20 for c in chunks)
    assert chunks[0].split()[-5:] == chunks[1].split()[:5]            # consecutive chunks share 5 words
    assert {w for c in chunks for w in c.split()} == set(words)       # nothing is lost


def test_clean_pdf_text_fixes_ligatures_and_hyphenation():
    assert ingest.clean_pdf_text("a classiﬁer is explain-\nable\nand  simple") == "a classifier is explainable and simple"


def test_note_chunks_one_per_section_with_heading(tmp_path):
    note = tmp_path / "demo.md"
    note.write_text("# Demo note\n\nIntro text.\n\n## First topic\n\nAlpha facts.\n\n## Second topic\n\nBeta facts.\n")
    chunks = ingest.note_chunks(note)
    assert [c.section for c in chunks] == ["Demo note", "First topic", "Second topic"]
    assert chunks[1].text.startswith("Demo note > First topic.")      # topic is part of the embedded text
    assert all(c.kind == "note" and c.citation == "Note: demo" for c in chunks)
    assert len({c.id for c in chunks}) == 3


def explanation(timestamp, flags=(), holiday=0):
    return {"timestamp": timestamp, "disagreements": [{"type": t, "severity": s} for t, s in flags],
            "lightgbm": {"shap": [{"feature": "is_holiday", "value": holiday}]}}


def test_christmas_eve_gets_a_date_query():
    qs = queries.queries_from_evidence(explanation("2014-12-24 18:00:00"))
    assert any("Christmas Eve" in q["query"] for q in qs)


def test_ordinary_day_gets_only_the_standing_queries():
    qs = queries.queries_from_evidence(explanation("2014-10-16 12:00:00"))
    assert [q["why"].split(":")[0] for q in qs] == ["always", "always"]


def test_flags_that_fired_add_queries_and_ok_flags_do_not():
    fired = queries.queries_from_evidence(explanation("2014-10-16 12:00:00", [("lime_fit", "warning"), ("shap_vs_lime", "high")]))
    quiet = queries.queries_from_evidence(explanation("2014-10-16 12:00:00", [("lime_fit", "ok"), ("shap_vs_lime", "ok")]))
    assert len(fired) == len(quiet) + 2
    assert any("LIME" in q["query"] for q in fired)


def test_public_holiday_query_when_holiday_flag_set():
    qs = queries.queries_from_evidence(explanation("2014-08-15 12:00:00", holiday=1))
    assert any("public holiday" in q["query"] for q in qs)


def test_dst_night_detected():
    assert queries._is_dst_change_night(__import__("pandas").Timestamp("2014-10-26 01:00"))
    assert not queries._is_dst_change_night(__import__("pandas").Timestamp("2014-10-26 12:00"))


def test_labels_and_context_format():
    hits = label_hits([{"citation": "Lundberg & Lee 2017 (SHAP)", "page": 4, "text": "abc"},
                       {"citation": "Note: x", "page": 0, "text": "def"}])
    assert [h["sid"] for h in hits] == ["S1", "S2"]
    assert format_context(hits) == "[S1] Lundberg & Lee 2017 (SHAP), p.4: abc\n\n[S2] Note: x: def"


def test_retrieval_quality_on_the_real_index():
    """Skipped when the index has not been built (run scripts/build_index.py)."""
    from forecast_explainer.rag.evaluate import evaluate
    try:
        _, summary = evaluate(k=5)
    except RuntimeError:
        pytest.skip("knowledge index not built")
    assert summary["recall_at_k"] >= 0.9
