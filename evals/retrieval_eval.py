"""
Eval 1: Retrieval. Does searching the notes find the right page?

Uses 20 hand-written questions (worded differently from the notes) and 5 questions
that are NOT in the notes. Also tests the Phase 1 choice of 120-word chunks
against smaller and bigger chunks.

Run from the project folder (needs Ollama running, takes about a minute):
    python evals/retrieval_eval.py
"""
import json
import time

import chromadb

from common import GOLD, ROOT, mean, pct, progress, save_result, wilson
import rag

CHUNK_SIZES = [(60, 12), (120, 25), (250, 50)]   # (words per chunk, overlap); 120 is the app's choice


def build_index(pages, size, overlap):
    """A throwaway, in-memory copy of the vector store with a different chunk size."""
    client = chromadb.EphemeralClient()
    col = client.create_collection(f"chunks_{size}", metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for page, text in pages:
        for j, chunk in enumerate(rag.chunk_page(text, size, overlap)):
            ids.append(f"p{page}_c{j}")
            docs.append(chunk)
            metas.append({"page": page})
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=rag.embed(docs))
    return col, len(docs)


def evaluate(col, gold, question_vectors, k=3):
    hits1 = hits3 = 0
    reciprocal_ranks, best_in, best_out, misses = [], [], [], []
    for item, vec in zip(gold["questions"], question_vectors["in"]):
        res = col.query(query_embeddings=[vec], n_results=k)
        pages = [m["page"] for m in res["metadatas"][0]]
        best_in.append(round(1 - res["distances"][0][0], 3))
        hits1 += pages[0] == item["page"]
        hits3 += item["page"] in pages
        rank = pages.index(item["page"]) + 1 if item["page"] in pages else None
        reciprocal_ranks.append(1 / rank if rank else 0)
        if pages[0] != item["page"]:
            misses.append({"question": item["q"], "expected": item["page"], "got": pages})
    for vec in question_vectors["out"]:
        res = col.query(query_embeddings=[vec], n_results=1)
        best_out.append(round(1 - res["distances"][0][0], 3))
    n = len(gold["questions"])
    return {
        "hit_at_1": pct(hits1, n), "hit_at_1_range": wilson(hits1, n),
        "hit_at_3": pct(hits3, n),
        "mrr": round(sum(reciprocal_ranks) / n, 3),
        "in_scope_best_score": {"min": min(best_in), "mean": mean(best_in)},
        "out_of_scope_best_score": {"max": max(best_out), "mean": mean(best_out)},
        # If every real question scores above every off-topic one, a single cut-off can tell them apart.
        "separable": min(best_in) > max(best_out),
        "misses": misses,
    }


def main():
    gold = json.loads((GOLD / "retrieval.json").read_text(encoding="utf-8"))
    pdf = ROOT / "sample_notes" / f"{gold['notes']}.pdf"
    print(f"Retrieval eval on {pdf.name} with {rag.EMBED_MODEL}")
    pages = rag.read_pdf(str(pdf))
    t0 = time.time()
    vectors = {"in": rag.embed([q["q"] for q in gold["questions"]]),
               "out": rag.embed(gold["out_of_scope"])}
    embed_ms = round(1000 * (time.time() - t0) / (len(vectors["in"]) + len(vectors["out"])))

    results = []
    for i, (size, overlap) in enumerate(CHUNK_SIZES, 1):
        col, n_chunks = build_index(pages, size, overlap)
        r = evaluate(col, gold, vectors)
        r.update(chunk_words=size, overlap=overlap, chunks=n_chunks, used_by_app=size == rag.CHUNK_WORDS)
        results.append(r)
        progress(i, len(CHUNK_SIZES), "chunk sizes tested")

    print()
    print(f"{'chunk words':>11} {'chunks':>7} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}  separable")
    for r in results:
        mark = "  <- app" if r["used_by_app"] else ""
        print(f"{r['chunk_words']:>11} {r['chunks']:>7} {r['hit_at_1']:>6} {r['hit_at_3']:>6} {r['mrr']:>6}  {r['separable']}{mark}")
    app = next(r for r in results if r["used_by_app"])
    if app["misses"]:
        print("\nQuestions the app's setting got wrong (top result):")
        for m in app["misses"]:
            print(f"  page {m['expected']} expected, got {m['got']}: {m['question']}")

    path = save_result("retrieval", {
        "notes": gold["notes"], "embed_model": rag.EMBED_MODEL, "questions": len(gold["questions"]),
        "out_of_scope": len(gold["out_of_scope"]), "ms_per_embedding": embed_ms, "by_chunk_size": results,
    })
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()