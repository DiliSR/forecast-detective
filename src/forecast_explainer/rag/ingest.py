"""Ingest: turn the knowledge files into chunks, embed them, and store them in a Chroma index.

Pipeline:  file -> text -> chunks (with metadata) -> embeddings -> Chroma collection on disk.

Why chunk at all? Retrieval returns whole chunks, and the LLM reads them. A whole paper is too long
and too unspecific; one sentence loses context. About 200 words is a paragraph or two: enough to be
self-explanatory, short enough that the embedding represents ONE idea.

Run via `python scripts/build_index.py`; notebooks/04 walks through it step by step.
"""

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from forecast_explainer.rag import COLLECTION, EMBED_MODEL, INDEX_DIR, NOTES_DIR, PAPERS, PAPERS_DIR

CHUNK_WORDS = 200      # target chunk length
OVERLAP_WORDS = 40     # words shared by consecutive chunks of the same page/section
MIN_WORDS = 30         # shorter fragments (page numbers, figure captions, empty pages) are dropped


@dataclass
class Chunk:
    """One retrievable piece of text plus everything needed to cite it.

    id:       unique and stable ("shap_lundberg2017:p4:0"); lets us dedupe and re-index.
    text:     the chunk itself (for notes this starts with "<note title> > <section>" so the topic is in the embedding).
    source:   file stem ("calendar_effects"), used for filtering and for the retrieval evaluation.
    citation: the label the LLM cites ("Lundberg & Lee 2017 (SHAP)" or "Note: calendar_effects").
    kind:     "paper" or "note"; lets a query ask only for domain notes or only for papers.
    page:     PDF page number (1-based) for papers, 0 for notes.
    section:  the "##" heading for notes, "" for papers.
    """
    id: str
    text: str
    source: str
    citation: str
    kind: str
    page: int = 0
    section: str = ""


# ----------------------------------------------------------------------------- chunking

def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Split text into windows of about `size` words, each sharing `overlap` words with the previous one.

    size:    window length in words (words, not characters or tokens, keeps the code simple and the
             chunk size independent of the tokenizer).
    overlap: why overlap? A sentence cut at a boundary would otherwise be half in one chunk and half in
             the next, and neither half retrieves well. With overlap, every sentence is whole in some chunk.
    Text that fits in one window is returned unchanged.
    """
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = size - overlap
    return [" ".join(words[start:start + size]) for start in range(0, len(words) - overlap, step)]


def clean_pdf_text(raw: str) -> str:
    """Repair what pypdf gets wrong: ligatures (fi), words hyphenated across lines, hard line breaks.

    NFKC normalisation turns ligature characters like "ﬁ" into "fi" so "classiﬁer" matches "classifier".
    """
    text = unicodedata.normalize("NFKC", raw)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)   # "explain-\nable" -> "explainable"
    return re.sub(r"\s+", " ", text).strip()


REFERENCES = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)


def paper_chunks(path: Path) -> list[Chunk]:
    """Chunk a PDF page by page. Chunks never cross a page, so every chunk has an exact page to cite.

    Ingestion stops at the "References" heading: reference lists are mostly author names and years, they
    match queries by accident and add no explanation. (The cost: appendices after the references are not indexed.)
    """
    label = PAPERS.get(path.stem, path.stem)
    chunks, stop = [], False
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        raw = page.extract_text() or ""
        match = REFERENCES.search(raw)
        if match:
            raw, stop = raw[:match.start()], True
        for j, text in enumerate(chunk_words(clean_pdf_text(raw))):
            if len(text.split()) >= MIN_WORDS:
                chunks.append(Chunk(f"{path.stem}:p{page_no}:{j}", text, path.stem, label, "paper", page=page_no))
        if stop:
            break
    return chunks


def note_chunks(path: Path) -> list[Chunk]:
    """Chunk a markdown note by "##" section: one section = one self-contained fact, so it becomes one chunk.

    The note title (first "# " line) and the section heading are put in front of the text. An embedding of
    "the load drops 29-43% at 20:00" alone does not mention Christmas Eve; the heading does.
    Sections longer than CHUNK_WORDS are split further with chunk_words.
    """
    title, sections = path.stem, []
    for block in re.split(r"^## ", path.read_text(), flags=re.MULTILINE):
        first, _, body = block.partition("\n")
        if first.startswith("# "):               # the part before the first "##": holds the note title
            title = first[2:].strip()
            body = body.strip()
            if body:
                sections.append((title, body))
        elif body.strip():
            sections.append((first.strip(), body.strip()))

    chunks = []
    for section, body in sections:
        for j, text in enumerate(chunk_words(re.sub(r"\s+", " ", body))):
            chunks.append(Chunk(f"{path.stem}:{re.sub(r'[^a-z0-9]+', '-', section.lower()).strip('-')}:{j}",
                                f"{title} > {section}. {text}", path.stem, f"Note: {path.stem}", "note", section=section))
    return chunks


def all_chunks() -> list[Chunk]:
    """Every chunk from knowledge/papers/*.pdf and knowledge/notes/*.md."""
    chunks = []
    for path in sorted(PAPERS_DIR.glob("*.pdf")):
        chunks += paper_chunks(path)
    for path in sorted(NOTES_DIR.glob("*.md")):
        chunks += note_chunks(path)
    return chunks


# ----------------------------------------------------------------------------- embedding + index

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """The embedding model, loaded once per process (loading takes a few seconds)."""
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]):
    """Texts -> unit-length vectors, shape (n, 384).

    normalize_embeddings=True makes every vector length 1, so cosine similarity is a plain dot product
    and Chroma's cosine distance = 1 - similarity.
    """
    return get_embedder().encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)


def get_collection(create: bool = False):
    """The Chroma collection stored in data/index (a persistent on-disk database).

    create=True makes the collection if it does not exist; otherwise a missing index raises, with a hint.
    The index uses cosine distance, the usual choice for sentence embeddings.
    """
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    if create:
        return client.get_or_create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}})
    try:
        return client.get_collection(COLLECTION)
    except Exception as exc:
        raise RuntimeError("knowledge index not found - run `python scripts/build_index.py`") from exc


def build_index(chunks: list[Chunk] | None = None, rebuild: bool = True) -> dict:
    """Embed the chunks and write them to the index. Returns simple statistics.

    chunks:  default = everything under knowledge/.
    rebuild: True deletes the old collection first, so the index always equals the current files
             (no stale chunks from a note that was edited or removed). The corpus is small enough
             that rebuilding takes well under a minute.
    """
    chunks = chunks if chunks is not None else all_chunks()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if rebuild:
        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        if COLLECTION in [c.name for c in client.list_collections()]:
            client.delete_collection(COLLECTION)
    collection = get_collection(create=True)

    for start in range(0, len(chunks), 256):          # batches keep memory flat
        batch = chunks[start:start + 256]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embed([c.text for c in batch]).tolist(),
            metadatas=[{"source": c.source, "citation": c.citation, "kind": c.kind,
                        "page": c.page, "section": c.section} for c in batch],
        )
    by_source: dict[str, int] = {}
    for c in chunks:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    return {"chunks": len(chunks), "by_source": by_source}
