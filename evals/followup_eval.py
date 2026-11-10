"""
Eval 4: Follow-up questions. Does the leak guard stop follow-ups that give the answer away?

For every question in the frozen exam set, the student "answers" with key point 1 only,
so point 2 is missing. Both examiners then write a follow-up about point 2.
Measured per examiner:
  - how often the first follow-up passes the leak guard
  - how often every try leaks and the generic fallback is used
  - how many answer words the rejected follow-ups contained

Every follow-up written (accepted AND rejected) goes to the review page, where you
judge "does this give the answer away?" without seeing the guard's decision.
python evals/report.py then tells you how often the guard agrees with you.

Run from the project folder (needs Ollama, about 2-4 minutes; safe to stop and rerun):
    python evals/followup_eval.py
"""
from common import Cache, ROOT, load_result, mean, pct, progress, save_result, short_hash
import examiner


def main():
    exam = load_result("exam_set")
    if not exam:
        print("Build the exam set first: python evals/exam_set.py")
        return
    items = [q for q in exam["items"] if len(q["key_points"]) >= 2]
    version = short_hash(examiner.FOLLOW_UP_PROMPT, examiner.LLM_MODEL,
                         *[p["style"] for p in examiner.PERSONAS.values()])
    cache = Cache(f"followup_{version}")

    cases = []
    jobs = [(q, key) for q in items for key in examiner.PERSONAS]
    print(f"Writing follow-ups for {len(items)} questions x {len(examiner.PERSONAS)} examiners "
          f"(prompt version {version})")
    for i, (q, persona_key) in enumerate(jobs, 1):
        persona = examiner.PERSONAS[persona_key]
        answer = q["key_points"][0]["evidence"]       # the student said point 1 only
        target = q["key_points"][1]                    # so the examiner probes point 2
        key = f"{q['id']}|{persona_key}"
        fu = cache.get(key)
        if fu is None or fu.get("model_failed") or (fu["fallback"] and not fu["tries"]):
            # (the second check also throws away bad results saved by older versions of this script)
            fu = examiner.make_follow_up(q, answer, target, persona)
            if fu.get("model_failed"):
                print(f"\nThe model couldn't run, so nothing was saved for {q['id']}. "
                      "Fix the memory problem and run this script again; finished items are kept.")
                return
            cache.put(key, fu)
        tries = []
        for t_i, t in enumerate(fu["tries"]):
            accepted = (not fu["fallback"]) and t_i == len(fu["tries"]) - 1
            tries.append({"id": f"{q['id']}-{persona_key}-{t_i + 1}", "question": t["question"],
                          "leaked": t["leaked"], "accepted": accepted})
        cases.append({
            "id": f"{q['id']}-{persona_key}", "exam_id": q["id"], "persona": persona_key,
            "question": q["question"], "student_answer": answer, "missing_point": target["point"],
            "evidence": target["evidence"], "final": fu["question"], "fallback": fu["fallback"],
            "tries": tries,
        })
        progress(i, len(jobs), "follow-ups")

    summary = {}
    for persona_key, persona in examiner.PERSONAS.items():
        cs = [c for c in cases if c["persona"] == persona_key]
        rejected = [t for c in cs for t in c["tries"] if not t["accepted"]]
        summary[persona_key] = {
            "name": persona["name"],
            "hint_words_allowed": persona["hint_words"],
            "cases": len(cs),
            "first_try_accepted": pct(sum(c["tries"] and c["tries"][0]["accepted"] for c in cs), len(cs)),
            "fallback_used": pct(sum(c["fallback"] for c in cs), len(cs)),
            "tries_rejected": len(rejected),
            "avg_leaked_words_when_rejected": mean([len(t["leaked"]) for t in rejected]),
            "most_leaked_words": sorted({w for t in rejected for w in t["leaked"]},
                                        key=lambda w: -sum(w in t["leaked"] for t in rejected))[:8],
        }

    print(f"\n{'examiner':18} {'hints ok':>8} {'1st try ok':>10} {'fallback':>9} {'rejected':>9}")
    for s in summary.values():
        print(f"{s['name']:18} {s['hint_words_allowed']:>8} {s['first_try_accepted']:>9}% "
              f"{s['fallback_used']:>8}% {s['tries_rejected']:>9}")
    examples = [t for c in cases for t in c["tries"] if t["leaked"] and not t["accepted"]][:3]
    if examples:
        print("\nExamples the guard rejected:")
        for t in examples:
            print(f"  \"{t['question']}\"  (gave away: {', '.join(t['leaked'])})")

    path = save_result("followup", {"model": examiner.LLM_MODEL, "prompt_version": version,
                                    "summary": summary, "cases": cases})
    print(f"\nSaved {path.relative_to(ROOT)}. Label these on the review page (follow-ups).")


if __name__ == "__main__":
    try:
        main()
    finally:
        examiner.unload()     # don't leave the model in memory for 30 minutes