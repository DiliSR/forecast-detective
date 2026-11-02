"""RAG (retrieval-augmented generation) knowledge base for the explanation agent.

What RAG does here: the explainers say WHAT drove a forecast; this knowledge base supplies the background
that interprets those numbers (how to read SHAP vs LIME, why Christmas Eve behaves like a holiday) together
with a source the explanation can cite. It supplies no numbers about the forecast itself - those always come
from the explainers.

    knowledge/papers/*.pdf  ┐  ingest.py  (chunk, embed)   ┐  retrieve.py  (query -> top chunks)
    knowledge/notes/*.md    ┘ ────────────────────────────► │ ──────────────────────────────────►  [S1] [S2] ...
                                                            data/index (Chroma)

    queries.py builds the queries from the explanation's evidence and disagreement flags.
"""

from forecast_explainer import DATA_DIR, PROJECT_ROOT

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
PAPERS_DIR = KNOWLEDGE_DIR / "papers"
NOTES_DIR = KNOWLEDGE_DIR / "notes"
INDEX_DIR = DATA_DIR / "index"

COLLECTION = "knowledge"

# A small sentence-embedding model (384 dimensions, ~90 MB, runs on CPU). Chosen because the knowledge base
# is tiny (a few hundred chunks) so a bigger model would only add download time and memory.
EMBED_MODEL = "all-MiniLM-L6-v2"

# Chunks farther than this (cosine distance, 0 = identical, 1 = unrelated) are not used. Measured on this
# knowledge base: the best chunk for 18 on-topic test queries was at most 0.55 away, for 8 off-topic queries
# (bread, football, ...) at least 0.62. 0.58 sits in that gap. The margin is thin (a small sample), so
# re-measure (notebook 04, step 6) after adding documents or changing EMBED_MODEL.
MAX_DISTANCE = 0.58

# Human-readable citation for each paper (file stem -> label). This label is what the LLM cites and what
# the app shows, so it must identify the paper, not the file name.
PAPERS = {
    "shap_lundberg2017": "Lundberg & Lee 2017 (SHAP)",
    "treeshap_lundberg2019": "Lundberg et al. 2020 (TreeSHAP)",
    "lime_ribeiro2016": "Ribeiro et al. 2016 (LIME)",
    "integrated_gradients_sundararajan2017": "Sundararajan et al. 2017 (Integrated Gradients)",
    "timeshap_bento2021": "Bento et al. 2021 (TimeSHAP)",
}
