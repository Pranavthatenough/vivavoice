"""
VivaVoice - human labelling for the evaluation (the review page, /review.html).

Three kinds of items to judge:
  grading    real answers from your vivas: which key points did the answer cover?
  grounding  questions from the frozen exam set: answerable from the passage? right level?
  leak       follow-ups from evals/followup_eval.py: do they give the missing point away?

The examiner's own verdicts are never sent to the page. Labelling "blind" stops
the AI's answer from nudging yours, so your labels are a fair test of it.
Labels are saved in evals/labels.json, your hand-labelled dataset.
"""
import json
import time

from rag import ROOT

EVALS = ROOT / "evals"
RESULTS = EVALS / "results"
LABELS = EVALS / "labels.json"
SESSIONS = ROOT / "data" / "sessions"
LEVEL_NAMES = ["recall", "explain", "apply"]


def _read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def model_points(q):
    """
    The examiner's verdict on the FIRST answer, as [(point, covered), ...].
    Handles sessions saved by every phase: Phase 2 stored points as plain strings.
    """
    if q.get("first"):
        return [(p["point"], bool(p["covered"])) for p in q["first"]["key_points"]]
    kps = q.get("key_points") or []
    if kps and isinstance(kps[0], str):
        covered = set(q.get("covered") or [])
        return [(p, p in covered) for p in kps]
    return [(p["point"], bool(p.get("covered", p.get("status") == "covered"))) for p in kps]


def items():
    out = []
    for path in sorted(SESSIONS.glob("*.json")) if SESSIONS.exists() else []:
        data = _read(path)
        for q in data.get("questions", []):
            answer = (q.get("answer") or "").strip()
            points = [p for p, _ in model_points(q)]
            if not points or answer.lower().rstrip(".!") in ("", "pass", "skip", "idk"):
                continue
            out.append({
                "id": f"grade/{path.stem}/{q.get('number', 0)}", "type": "grading",
                "question": q.get("question", ""), "answer": answer, "points": points,
                "excerpt": q.get("source_text", ""), "page": q.get("page"),
                "language": data.get("language", "en"),
            })
    for q in _read(RESULTS / "exam_set.json").get("items", []):
        out.append({
            "id": f"ground/{q['id']}", "type": "grounding", "question": q["question"],
            "level": LEVEL_NAMES[q["level"]], "excerpt": q["source_text"], "page": q["page"],
        })
    for c in _read(RESULTS / "followup.json").get("cases", []):
        for t in c["tries"]:
            out.append({
                "id": f"leak/{t['id']}", "type": "leak", "question": c["question"],
                "answer": c["student_answer"], "missing": c["missing_point"], "follow_up": t["question"],
            })
    return out


def load_labels():
    return _read(LABELS)


def save_label(item_id, value):
    labels = load_labels()
    if value is None:
        labels.pop(item_id, None)
    else:
        labels[item_id] = {"value": value, "at": time.strftime("%Y-%m-%d %H:%M")}
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    LABELS.write_text(json.dumps(labels, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return len(labels)