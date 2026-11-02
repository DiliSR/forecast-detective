"""Retrieve: run queries against the index and return labelled, deduplicated chunks.

A "hit" is a dict:
    id, text, source, citation, kind, page, section   (from ingest.Chunk)
    distance   cosine distance to the best-matching query (0 = identical direction, 1 = unrelated; lower is better)
    queries    which of the queries retrieved it (for debugging: "why did this chunk appear?")
    sid        "S1", "S2", ... assigned by label_hits; this is the ID the LLM cites.
"""

from forecast_explainer.rag import MAX_DISTANCE
from forecast_explainer.rag.ingest import embed, get_collection
from forecast_explainer.rag.queries import queries_from_evidence


def search(queries: list[str], k: int = 3, kind: str | None = None, max_total: int = 10,
           max_distance: float | None = None, guarantee_best: bool = True) -> list[dict]:
    """Best chunks for a list of queries, merged and deduplicated.

    queries:        query strings. Each is embedded and searched separately, because one query per topic
                    retrieves better than one long query mixing topics.
    k:              how many chunks to take per query. Several queries x k means the same chunk is often
                    found twice; duplicates are merged, keeping the smallest distance.
    kind:           "note" or "paper" to search only domain notes or only papers (None = both).
    max_total:      keep at most this many chunks overall, best first. This bounds the LLM prompt size.
    max_distance:   drop chunks whose distance is above this (None = keep all). A chunk that is the
                    "least bad" match for a query is still unrelated; better to return nothing than to
                    hand the LLM text it might feel obliged to cite.
    guarantee_best: reserve a slot for each query's own best chunk before filling the rest by distance.
                    Without it, generic queries that match many chunks closely (e.g. "how to read SHAP")
                    crowd out the one chunk that answers a specific flag (e.g. LIME's poor fit).
    """
    collection = get_collection()
    where = {"kind": kind} if kind else None
    found: dict[str, dict] = {}
    best_per_query: list[str] = []
    for query, vector in zip(queries, embed(queries)):
        result = collection.query(query_embeddings=[vector.tolist()], n_results=k, where=where)
        ids = result["ids"][0]
        for id_, text, meta, dist in zip(ids, result["documents"][0], result["metadatas"][0], result["distances"][0]):
            hit = found.setdefault(id_, {"id": id_, "text": text, **meta, "distance": dist, "queries": []})
            hit["distance"] = min(hit["distance"], dist)
            hit["queries"].append(query)
        if ids:
            best_per_query.append(ids[0])

    def allowed(hit: dict) -> bool:
        return max_distance is None or hit["distance"] <= max_distance

    ranked = sorted((h for h in found.values() if allowed(h)), key=lambda h: h["distance"])
    chosen: dict[str, dict] = {}
    if guarantee_best:
        for hit in ranked:
            if hit["id"] in best_per_query and len(chosen) < max_total:
                chosen[hit["id"]] = hit
    for hit in ranked:                       # fill the remaining slots with the next-best chunks
        if len(chosen) < max_total:
            chosen.setdefault(hit["id"], hit)
    return sorted(chosen.values(), key=lambda h: h["distance"])


def label_hits(hits: list[dict]) -> list[dict]:
    """Give each hit a citation ID S1, S2, ... in order of relevance (best = S1)."""
    return [{**h, "sid": f"S{i}"} for i, h in enumerate(hits, start=1)]


def format_context(hits: list[dict]) -> str:
    """The retrieved chunks as one text block for an LLM prompt: "[S1] <citation>[, p.N]: <text>".

    The LLM is told to cite these IDs; the judge later checks each cited claim against the chunk text.
    """
    lines = []
    for h in hits:
        where = f", p.{h['page']}" if h["page"] else ""
        lines.append(f"[{h['sid']}] {h['citation']}{where}: {h['text']}")
    return "\n\n".join(lines)


def retrieve_for_explanation(explanation: dict, k: int = 3, max_total: int = 10,
                             max_distance: float | None = MAX_DISTANCE, models="both") -> tuple[list[dict], list[dict]]:
    """The full retrieval step for one explanation: build queries from its evidence, search, label.

    Returns (hits, queries). `queries` is returned too so the app can show what was searched and why.
    """
    queries = queries_from_evidence(explanation, models)
    hits = search([q["query"] for q in queries], k=k, max_total=max_total, max_distance=max_distance)
    return label_hits(hits), queries
