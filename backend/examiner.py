"""
VivaVoice - the examiner: grounded questions, checklist grading, viva sessions.

Terminal viva (run from the project folder, after ingesting the notes):
    python backend/examiner.py physics_current
"""
import json
import random
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from difflib import SequenceMatcher

import ollama
from pydantic import BaseModel

from rag import ROOT, get_db

LLM_MODEL = "llama3.2:3b"
NUM_QUESTIONS = 5
PASS_MARK = 4          # scores below this go into "pages to revise"
MAX_TRIES = 3          # attempts to get a question whose evidence checks out
SESSIONS_DIR = ROOT / "data" / "sessions"


# ---------- What we want the LLM to return (forces clean JSON) ----------

class KeyPoint(BaseModel):
    point: str
    evidence: str      # a short quote copied word for word from the notes


class Question(BaseModel):
    question: str
    key_points: list[KeyPoint]


class PointCheck(BaseModel):
    point: str
    covered: bool


class Grade(BaseModel):
    checks: list[PointCheck]
    feedback: str


def ask_llm(system, user, schema, temperature):
    """Send a prompt to the local LLM and get back a validated object."""
    reply = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        format=schema.model_json_schema(),
        options={"temperature": temperature},
    )
    return schema.model_validate_json(reply["message"]["content"])


# ---------- Grounding check: is the evidence really in the notes? ----------

def find_quote(quote, source):
    """
    Look for the quote inside the notes excerpt.
    Returns (start, end) of the match in `source`, or None if it isn't there.
    Small differences (a changed word ending, punctuation) are tolerated:
    at least 80% of the quote must appear as one unbroken run of text.
    """
    q = re.sub(r"\s+", " ", quote).strip(" \"'.,;:").lower()
    if len(q.split()) < 3:
        return None                              # too short to prove anything
    src = source.lower()
    i = src.find(q)
    if i >= 0:
        return (i, i + len(q))
    m = SequenceMatcher(None, src, q, autojunk=False).find_longest_match(0, len(src), 0, len(q))
    if m.size >= 0.8 * len(q):
        return (m.a, m.a + m.size)
    return None


# ---------- Question generator ----------

QUESTION_PROMPT = """You are a university viva examiner.
Read the notes excerpt and ask ONE oral viva question about ONE idea in it.
Rules:
- The full answer must be in the excerpt. Never use outside knowledge.
- Stay on a single topic. Do not combine ideas from different parts of a subject.
- A good student should be able to answer out loud in 2 to 4 sentences.
- Prefer "why" and "explain" questions over plain definitions.
- Never ask for a derivation or a long equation.
Then give 2 to 4 key points that a complete answer must contain.
For each key point, copy the evidence: 5 to 20 words copied EXACTLY, word for word,
from the excerpt, that prove the point. Do not change or shorten words inside the quote."""


def pick_chunks(name, n=NUM_QUESTIONS):
    """Choose n chunks to ask about, spreading them across different pages."""
    data = get_db().get_collection(name).get(include=["documents", "metadatas"])
    chunks = [{"id": i, "text": d, "page": m["page"]}
              for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])]
    all_pages = sorted({c["page"] for c in chunks})
    usable = [c for c in chunks if len(c["text"].split()) >= 40]   # skip scraps
    random.shuffle(usable)
    picked, pages = [], set()
    for c in usable:                         # first pass: one chunk per page
        if c["page"] not in pages:
            picked.append(c)
            pages.add(c["page"])
    for c in usable:                         # second pass: fill up if few pages
        if c not in picked:
            picked.append(c)
    return picked[:n], all_pages


def make_question(chunk):
    """
    Generate a question and check every key point's evidence against the notes.
    If any quote can't be found, the question may be made up, so try again.
    """
    t0 = time.time()
    best = None
    for attempt in range(1, MAX_TRIES + 1):
        q = ask_llm(QUESTION_PROMPT,
                    f"Notes excerpt (page {chunk['page']}):\n{chunk['text']}",
                    Question, temperature=0.7)
        points = [{"point": kp.point, "evidence": kp.evidence,
                   "span": find_quote(kp.evidence, chunk["text"])} for kp in q.key_points]
        found = sum(p["span"] is not None for p in points)
        candidate = {"question": q.question, "key_points": points,
                     "grounded": bool(points) and found == len(points), "attempts": attempt}
        if best is None or found > sum(p["span"] is not None for p in best["key_points"]):
            best = candidate
        if candidate["grounded"]:
            best = candidate
            break
    best.update(page=chunk["page"], source_text=chunk["text"], attempts=attempt,
                seconds_to_generate=round(time.time() - t0, 1))
    return best


# ---------- Grader ----------

GRADE_PROMPT = """You are a fair but strict viva examiner grading a SPOKEN answer.
For each numbered key point, decide if the student's answer covers it.
Rules:
- Judge meaning, not exact words. Informal language and small spelling or speech-to-text errors are fine.
- A point is covered only if the answer states it correctly. Vague hints do not count.
- If the answer is wrong about a point, it is not covered.
- Use the notes excerpt as the only source of truth.
- Return the checks in the same order as the key points.
Then write feedback in 1 or 2 sentences, speaking directly to the student:
say what was good and name the most important missing point."""

IDK = {"", "idk", "pass", "skip", "i don't know", "i dont know", "dont know", "don't know", "no idea"}


def grade(q, answer):
    """Mark each key point covered or missed, then calculate the score in code."""
    if answer.strip().lower().rstrip(".!") in IDK:
        covered = [False] * len(q["key_points"])
        feedback = f"No answer given. Revise page {q['page']}."
    else:
        points = "\n".join(f"{i}. {p['point']}" for i, p in enumerate(q["key_points"], 1))
        g = ask_llm(GRADE_PROMPT,
                    f"Notes excerpt:\n{q['source_text']}\n\nQuestion: {q['question']}\n\n"
                    f"Key points:\n{points}\n\nStudent's answer: {answer}",
                    Grade, temperature=0)
        # Line the checks up with the key points by position; a missing check counts as missed.
        covered = [i < len(g.checks) and g.checks[i].covered for i in range(len(q["key_points"]))]
        feedback = g.feedback
    # The LLM only says yes or no per point. The score is calculated here,
    # so the same coverage always gets the same score.
    score = round(5 * sum(covered) / max(len(covered), 1))
    marked = [{**p, "covered": c} for p, c in zip(q["key_points"], covered)]
    return {"score": score, "key_points": marked, "feedback": feedback}


# ---------- A viva session ----------

class VivaSession:
    """
    One viva: a list of questions, asked one at a time.
    While you answer question N, question N+1 is already being written in the
    background (prefetching), so you rarely wait between questions.
    """

    def __init__(self, notes, n=NUM_QUESTIONS):
        self.id = uuid.uuid4().hex[:8]
        self.notes = notes
        self.chunks, self.all_pages = pick_chunks(notes, n)
        if not self.chunks:
            raise ValueError("These notes have no passages long enough to ask about.")
        self.items = []
        self.started = datetime.now()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._next = self._pool.submit(make_question, self.chunks[0])
        self._saved = None

    @property
    def total(self):
        return len(self.chunks)

    @property
    def done(self):
        return len(self.items) == self.total and "score" in self.items[-1]

    def next_question(self):
        """Return the next question; it is usually ready already."""
        if self.items and "score" not in self.items[-1]:
            return self.items[-1]               # current question not answered yet
        if len(self.items) >= self.total:
            return None
        t0 = time.time()
        q = self._next.result()                 # waits only if not ready yet
        q["seconds_waited"] = round(time.time() - t0, 1)
        q["number"] = len(self.items) + 1
        q["asked_at"] = time.time()
        self.items.append(q)
        i = len(self.items)
        self._next = self._pool.submit(make_question, self.chunks[i]) if i < self.total else None
        return q

    def answer(self, text):
        """Grade the answer to the current question."""
        if not self.items or "score" in self.items[-1]:
            raise ValueError("There is no question waiting for an answer.")
        q = self.items[-1]
        t0 = time.time()
        q["seconds_to_answer"] = round(t0 - q["asked_at"], 1)
        q["answer"] = text
        q.update(grade(q, text))
        q["seconds_to_grade"] = round(time.time() - t0, 1)
        return q

    def report(self):
        answered = [q for q in self.items if "score" in q]
        page_scores = {p: [] for p in self.all_pages}
        for q in answered:
            page_scores[q["page"]].append(q["score"])
        pages = [{"page": p, "score": round(sum(s) / len(s), 1) if s else None}
                 for p, s in page_scores.items()]
        report = {
            "session_id": self.id,
            "notes": self.notes,
            "model": LLM_MODEL,
            "total": sum(q["score"] for q in answered),
            "max": 5 * len(answered),
            "revise_pages": sorted({q["page"] for q in answered if q["score"] < PASS_MARK}),
            "pages": pages,
            "grounded_questions": sum(q["grounded"] for q in answered),
            "questions": [{k: v for k, v in q.items() if k != "asked_at"} for q in answered],
        }
        if self.done:
            self.save(report)
            report["saved_to"] = self._saved
        return report

    def save(self, report):
        """Save the finished session once; Phase 6 uses these files for evaluation."""
        if self._saved:
            return
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = SESSIONS_DIR / f"{self.notes}_{self.started:%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        self._saved = str(path.relative_to(ROOT))


# ---------- Terminal viva ----------

def run_viva(name):
    print(f"\nVivaVoice examiner | notes: {name} | model: {LLM_MODEL}")
    print(f"{NUM_QUESTIONS} questions. Type your answer and press Enter. Type 'pass' to skip.\n")
    s = VivaSession(name)
    while (q := s.next_question()) is not None:
        print(f"Q{q['number']}. {q['question']}")
        r = s.answer(input("> "))
        print(f"Score: {r['score']}/5. {r['feedback']}")
        missed = [p["point"] for p in r["key_points"] if not p["covered"]]
        if missed:
            print("Missed: " + "; ".join(missed))
        print()
    rep = s.report()
    print("=" * 50)
    print(f"Total: {rep['total']}/{rep['max']}")
    print("Revise pages: " + (", ".join(map(str, rep["revise_pages"])) or "none, well done!"))
    print(f"Grounded questions: {rep['grounded_questions']}/{len(rep['questions'])}")
    print(f"Session saved to {rep['saved_to']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
    else:
        run_viva(sys.argv[1])