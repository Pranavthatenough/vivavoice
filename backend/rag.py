"""
VivaVoice - Phase 1: notes -> chunks -> embeddings -> vector store.

Usage (run from the project folder):
    python backend/rag.py ingest sample_notes/physics_current.pdf
    python backend/rag.py search physics_current "What is drift velocity?"
"""
import re
import sys
import time
from pathlib import Path

import chromadb
import ollama
import pymupdf

EMBED_MODEL = "nomic-embed-text"   # embedding model, run by Ollama
ROOT = Path(__file__).resolve().parent.parent   # the project folder
DB_PATH = str(ROOT / "data" / "chroma")         # where the vector store is saved (ignored by git)
CHUNK_WORDS = 120               # target words per chunk
OVERLAP_WORDS = 25              # words repeated between neighbouring chunks


def read_pdf(pdf_path):
    """Return a list of (page_number, text) for every page that has text."""
    with pymupdf.open(pdf_path) as doc:
        raw = [[line.strip() for line in page.get_text().splitlines() if line.strip()] for page in doc]

    # Headers and footers repeat on most pages; they add noise, so drop them.
    counts = {}
    for lines in raw:
        for line in set(lines):
            counts[line] = counts.get(line, 0) + 1
    repeated = {line for line, n in counts.items() if len(raw) >= 3 and n > len(raw) / 2}

    pages = []
    for i, lines in enumerate(raw, start=1):
        kept = [l for l in lines if l not in repeated and not re.fullmatch(r"(Page\s*)?\d+", l)]
        text = re.sub(r"\s+", " ", " ".join(kept)).strip()
        if text:
            pages.append((i, text))
    if not pages:
        raise ValueError(f"No text found in {pdf_path}. Is it a scanned PDF?")
    return pages


def chunk_page(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Split one page's text into overlapping chunks of about `size` words."""
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


class EmbeddingUnavailable(Exception):
    """The embedding model couldn't run, even after retrying."""


_embed_on_cpu = False


def embed(texts):
    """
    Turn a list of texts into vectors using the local Ollama model.
    Like the examiner's LLM calls, it retries when memory is short, and moves
    to the CPU for the rest of the run if the GPU's own memory is full.
    """
    global _embed_on_cpu
    last = None
    for attempt in range(1, 4):
        try:
            options = {"num_gpu": 0} if _embed_on_cpu else {}
            return ollama.embed(model=EMBED_MODEL, input=texts, options=options)["embeddings"]
        except Exception as e:
            last = e
            print(f"Embedding failed (try {attempt} of 3): {str(e)[:160]}")
            if "cuda" in str(e).lower() and not _embed_on_cpu:
                _embed_on_cpu = True
                print("The GPU is out of memory: embedding on the CPU instead.")
                continue
            time.sleep(2 * attempt)
    raise EmbeddingUnavailable(
        f"The embedding model ({EMBED_MODEL}) couldn't run. It's usually short of memory: close other apps "
        f"and try again. Details: {str(last)[:200]}")


def collection_name(pdf_path):
    """Chroma needs simple names: letters, digits, _ . - (3 to 63 characters)."""
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(pdf_path).stem)[:60].strip("_-")
    return name if len(name) >= 3 else f"notes_{name}"


def get_db():
    return chromadb.PersistentClient(path=DB_PATH)


def ingest(pdf_path):
    """Read a PDF, chunk it, embed the chunks and save them with page numbers."""
    name = collection_name(pdf_path)
    ids, docs, metas = [], [], []
    for page, text in read_pdf(pdf_path):
        for j, chunk in enumerate(chunk_page(text)):
            ids.append(f"p{page}_c{j}")
            docs.append(chunk)
            metas.append({"page": page, "source": Path(pdf_path).name})

    # Embed BEFORE touching the database: if embedding fails, the old copy of the
    # notes (if any) is left untouched instead of being replaced by an empty one.
    vectors = embed(docs)
    db = get_db()
    if name in [c.name for c in db.list_collections()]:
        db.delete_collection(name)      # re-ingesting replaces the old copy
    col = db.create_collection(name, metadata={"hnsw:space": "cosine"})
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
    pages = len({m["page"] for m in metas})
    print(f"Saved {len(docs)} chunks from {pages} pages into collection '{name}'.")
    return name


def search(name, question, k=3):
    """Return the k chunks closest in meaning to the question."""
    col = get_db().get_collection(name)
    res = col.query(query_embeddings=embed([question]), n_results=k)
    return [
        {"page": m["page"], "score": round(1 - d, 3), "text": doc}
        for doc, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "ingest":
        ingest(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "search":
        for hit in search(sys.argv[2], " ".join(sys.argv[3:])):
            print(f"\n[page {hit['page']} | score {hit['score']}]\n{hit['text'][:200]}...")
    else:
        print(__doc__)