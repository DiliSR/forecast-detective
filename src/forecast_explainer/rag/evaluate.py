"""Evaluate retrieval on its own, before any LLM is involved.

Why separately? If the explanation is wrong, there are two possible causes: retrieval handed the LLM the wrong
chunks, or the LLM misused the right ones. Measuring retrieval alone isolates the first cause, and it is
fast, free and deterministic.

A test case is {"query": ..., "expect": [source, ...], "section": optional text}:
  - expect:  file stems (e.g. "calendar_effects"); retrieval succeeds if ANY of them appears in the top k.
  - section: optional; for notes, the matching chunk's section heading must also contain this text.
             It checks that Christmas Eve's query returns the Christmas Eve section, not just the same file.
Metrics:
  - recall@k: share of queries with a correct chunk among the top k.
  - MRR (mean reciprocal rank): average of 1/rank of the first correct chunk (0 if none). 1.0 = always first.
"""

import json
from pathlib import Path

import pandas as pd

from forecast_explainer import PROJECT_ROOT
from forecast_explainer.rag.retrieve import search

PAIRS_PATH = PROJECT_ROOT / "eval" / "retrieval_pairs.json"


def load_pairs(path: Path = PAIRS_PATH) -> list[dict]:
    """The hand-written query -> expected-source test cases."""
    return json.loads(path.read_text())


def _is_correct(hit: dict, case: dict) -> bool:
    """Does this chunk satisfy the test case (right source and, if required, right section)?"""
    if hit["source"] not in case["expect"]:
        return False
    return "section" not in case or case["section"].lower() in hit["section"].lower()


def evaluate(pairs: list[dict] | None = None, k: int = 3, kind: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Run every test case with `k` results per query.

    Returns (per-query table, summary dict). The table has the rank of the first correct chunk (NaN if none)
    and what was retrieved instead, so failures can be inspected.
    Each query is searched alone (not merged with others) so a hit's rank is the real search rank.
    """
    pairs = pairs if pairs is not None else load_pairs()
    rows = []
    for case in pairs:
        hits = search([case["query"]], k=k, kind=kind, max_total=k, guarantee_best=False)
        rank = next((i for i, h in enumerate(hits, start=1) if _is_correct(h, case)), None)
        rows.append({
            "query": case["query"], "expected": "/".join(case["expect"]), "rank_of_first_correct": rank,
            "top_hits": ", ".join(f"{h['source']}{':' + h['section'][:18] if h['section'] else ':p' + str(h['page'])}" for h in hits),
        })
    table = pd.DataFrame(rows)
    ranks = table["rank_of_first_correct"]
    summary = {
        "k": k, "n_queries": len(table),
        "recall_at_k": float(ranks.notna().mean()),
        "mrr": float((1 / ranks).fillna(0).mean()),
    }
    return table, summary
