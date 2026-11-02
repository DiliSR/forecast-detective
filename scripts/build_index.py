"""Build (or rebuild) the knowledge index from knowledge/papers and knowledge/notes.

    python scripts/build_index.py

Run it again whenever you add or edit a paper or note. The index is written to data/index/.
"""

import json
import warnings

from forecast_explainer.rag.ingest import build_index

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    stats = build_index()
    print(json.dumps(stats, indent=2))
    print(f"\nindexed {stats['chunks']} chunks")
