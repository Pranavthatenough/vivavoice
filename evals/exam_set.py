"""
Eval 2: Question generation. Are the examiner's questions grounded in the notes?

Builds a FROZEN exam set: 12 questions (4 recall, 4 explain, 4 apply) from passages
spread across the notes. It's saved to evals/results/exam_set.json and reused by the
grading and follow-up evals, so they always test the same questions, even after
you change a prompt. That makes before/after comparisons fair.

Run from the project folder (needs Ollama, about 2-5 minutes):
    python evals/exam_set.py             # build it, or finish an interrupted build
    python evals/exam_set.py --rebuild   # throw the old set away and make a new one
"""
import random
import re
import sys
import time

from common import Cache, ROOT, load_result, mean, pct, percentile, progress, save_result, short_hash, wilson
import examiner
from rag import get_db

NOTES = "physics_current"
PER_LEVEL = 4
SEED = 7           # fixed, so the same passages are picked every time

# A rough automatic check that a question matches its level. Your own labels on
# the review page are the real check; this just flags obvious mismatches.
LEVEL_PATTERNS = {
    "recall": r"^\s*(what|which|state|name|define|give)\b",
    "explain": r"\b(why|how|explain)\b",
    "apply": r"\b(if|would|happen|suppose|doubled|halved|increase|decrease)\b",
}


def pick_passages(n):
    data = get_db().get_collection(NOTES).get(include=["documents", "metadatas"])
    chunks = [{"id": i, "text": d, "page": m["page"]}
              for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
              if len(d.split()) >= 40]
    rng = random.Random(SEED)
    rng.shuffle(chunks)
    picked, pages_seen = [], {}
    # Spread across pages: take passages from the least-used page first.
    while chunks and len(picked) < n:
        chunks.sort(key=lambda c: pages_seen.get(c["page"], 0))
        c = chunks.pop(0)
        picked.append(c)
        pages_seen[c["page"]] = pages_seen.get(c["page"], 0) + 1
    return picked


def main():
    rebuild = "--rebuild" in sys.argv
    if load_result("exam_set") and not rebuild:
        print("exam_set.json already exists: reporting on it. Use --rebuild to make a new one.")
        items = load_result("exam_set")["items"]
    else:
        prompt_version = short_hash(examiner.QUESTION_PROMPT, *[l["rule"] for l in examiner.LEVELS])
        cache = Cache(f"exam_set_{prompt_version}")
        passages = pick_passages(PER_LEVEL * len(examiner.LEVELS))
        print(f"Generating {len(passages)} questions with {examiner.LLM_MODEL} (prompt version {prompt_version})")
        items = []
        for i, chunk in enumerate(passages):
            level = i % len(examiner.LEVELS)
            key = f"{chunk['id']}|{level}|{examiner.LLM_MODEL}"
            q = cache.get(key)
            if q is None:
                q = examiner.make_question(chunk, level)
                cache.put(key, q)
            q["id"] = f"x{i + 1:02d}"
            items.append(q)
            progress(i + 1, len(passages), "questions")

    rows = []
    for q in items:
        level = examiner.LEVELS[q["level"]]["name"]
        rows.append({
            "level": level,
            "grounded": q["grounded"],
            "first_try": q["grounded"] and q["attempts"] == 1,
            "attempts": q["attempts"],
            "points": len(q["key_points"]),
            "points_found": sum(p["span"] is not None for p in q["key_points"]),
            "level_match": bool(re.search(LEVEL_PATTERNS[level], q["question"], re.I)),
            "seconds": q["seconds_to_generate"],
        })

    def summarise(rs):
        n = len(rs)
        g = sum(r["grounded"] for r in rs)
        return {
            "questions": n,
            "grounded": pct(g, n), "grounded_range": wilson(g, n),
            "grounded_first_try": pct(sum(r["first_try"] for r in rs), n),
            "avg_attempts": mean([r["attempts"] for r in rs]),
            "quotes_found": pct(sum(r["points_found"] for r in rs), sum(r["points"] for r in rs)),
            "level_match_auto": pct(sum(r["level_match"] for r in rs), n),
            "seconds_median": percentile([r["seconds"] for r in rs], 0.5),
            "seconds_p90": percentile([r["seconds"] for r in rs], 0.9),
        }

    summary = {"all": summarise(rows)}
    for level in LEVEL_PATTERNS:
        summary[level] = summarise([r for r in rows if r["level"] == level])

    print(f"\n{'':8} {'grounded':>9} {'1st try':>8} {'attempts':>9} {'quotes ok':>10} {'level ok':>9} {'median s':>9}")
    for name, s in summary.items():
        print(f"{name:8} {s['grounded']:>8}% {s['grounded_first_try']:>7}% {s['avg_attempts']:>9} "
              f"{s['quotes_found']:>9}% {s['level_match_auto']:>8}% {s['seconds_median']:>9}")
    ungrounded = [q for q in items if not q["grounded"]]
    if ungrounded:
        print("\nStill not grounded after all tries:")
        for q in ungrounded:
            print(f"  {q['id']} (page {q['page']}): {q['question']}")

    path = save_result("exam_set", {
        "notes": NOTES, "model": examiner.LLM_MODEL,
        "prompt_version": short_hash(examiner.QUESTION_PROMPT, *[l["rule"] for l in examiner.LEVELS]),
        "built": time.strftime("%Y-%m-%d %H:%M"), "summary": summary, "items": items,
    })
    print(f"\nSaved {path.relative_to(ROOT)}. Label these on the review page (grounding).")


if __name__ == "__main__":
    try:
        main()
    finally:
        examiner.unload()     # don't leave the model in memory for 30 minutes