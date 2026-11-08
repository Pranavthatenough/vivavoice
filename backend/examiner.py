"""
VivaVoice - Phase 2: question generator + grader, in the terminal.

Usage (run from the project folder, after ingesting the notes):
    python backend/examiner.py physics_current
"""
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import ollama
from pydantic import BaseModel

from rag import get_db

LLM_MODEL = "llama3.2:3b"   # if this runs out of memory, use "llama3.2:3b"
NUM_QUESTIONS = 5
PASS_MARK = 4               # scores below this go into "pages to revise"


# ---------- What we want the LLM to return (forces clean JSON) ----------

class Question(BaseModel):
    question: str
    key_points: list[str]


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


# ---------- Question generator ----------

QUESTION_PROMPT = """You are a university viva examiner.
Using ONLY the notes excerpt you are given, ask ONE oral viva question.
Rules:
- The answer must be fully contained in the excerpt. Never use outside knowledge.
- A good student should be able to answer it out loud in 2 to 4 sentences.
- Prefer "why" and "explain" questions over pure definitions.
- Do not ask the student to write or derive long equations.
- Do not repeat any question listed under "Already asked".
Also list 2 to 4 key points, taken from the excerpt, that a complete answer must contain.
Each key point must be one short phrase."""


def pick_chunks(name, n=NUM_QUESTIONS):
    """Choose n chunks to ask about, spreading them across different pages."""
    data = get_db().get_collection(name).get(include=["documents", "metadatas"])
    chunks = [{"id": i, "text": d, "page": m["page"]}
              for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
              if len(d.split()) >= 40]          # skip scraps too short to ask about
    random.shuffle(chunks)
    picked, pages = [], set()
    for c in chunks:                            # first pass: one chunk per page
        if c["page"] not in pages:
            picked.append(c)
            pages.add(c["page"])
    for c in chunks:                            # second pass: fill up if few pages
        if c not in picked:
            picked.append(c)
    return picked[:n]


def make_question(chunk, already_asked):
    asked = "\n".join(f"- {q}" for q in already_asked) or "- (none)"
    q = ask_llm(QUESTION_PROMPT,
                f"Notes excerpt (page {chunk['page']}):\n{chunk['text']}\n\nAlready asked:\n{asked}",
                Question, temperature=0.7)
    return {"question": q.question, "key_points": q.key_points,
            "page": chunk["page"], "source_text": chunk["text"]}


# ---------- Grader ----------

GRADE_PROMPT = """You are a fair but strict viva examiner grading a SPOKEN answer.
For each key point, decide if the student's answer covers it.
Rules:
- Judge meaning, not exact words. Informal language and small spelling or speech-to-text errors are fine.
- A point is covered only if the answer states it correctly. Vague hints do not count.
- If the answer is wrong about a point, it is not covered.
- Use the notes excerpt as the only source of truth.
Then write feedback in 1 or 2 sentences, speaking directly to the student:
say what was good and name the most important missing point."""

IDK = {"", "idk", "pass", "skip", "i don't know", "i dont know", "dont know", "don't know", "no idea"}


def grade(q, answer):
    """Return score 0-5 plus which key points were covered."""
    if answer.strip().lower().rstrip(".!") in IDK:
        return {"score": 0, "covered": [], "missed": q["key_points"],
                "feedback": f"No answer given. Revise page {q['page']}."}

    points = "\n".join(f"- {p}" for p in q["key_points"])
    g = ask_llm(GRADE_PROMPT,
                f"Notes excerpt:\n{q['source_text']}\n\nQuestion: {q['question']}\n\n"
                f"Key points:\n{points}\n\nStudent's answer: {answer}",
                Grade, temperature=0)

    covered = [c.point for c in g.checks if c.covered]
    missed = [c.point for c in g.checks if not c.covered]
    # The score is calculated from the checklist, not guessed by the LLM.
    # This makes grading consistent: the same coverage always gets the same score.
    score = round(5 * len(covered) / max(len(g.checks), 1))
    return {"score": score, "covered": covered, "missed": missed, "feedback": g.feedback}


# ---------- Terminal viva ----------

def run_viva(name):
    print(f"\nVivaVoice examiner | notes: {name} | model: {LLM_MODEL}")
    print(f"{NUM_QUESTIONS} questions. Type your answer and press Enter. Type 'pass' to skip.\n")

    session, asked = [], []
    for i, chunk in enumerate(pick_chunks(name), start=1):
        t0 = time.time()
        q = make_question(chunk, asked)
        asked.append(q["question"])
        print(f"Q{i}. {q['question']}")

        t1 = time.time()
        answer = input("> ")
        t2 = time.time()
        result = grade(q, answer)

        print(f"Score: {result['score']}/5. {result['feedback']}")
        if result["missed"]:
            print("Missed: " + "; ".join(result["missed"]))
        print()
        session.append({**q, "answer": answer, **result,
                        "seconds_to_generate": round(t1 - t0, 1),
                        "seconds_to_grade": round(time.time() - t2, 1)})

    total = sum(r["score"] for r in session)
    weak_pages = sorted({r["page"] for r in session if r["score"] < PASS_MARK})
    print("=" * 50)
    print(f"Total: {total}/{5 * len(session)}")
    print("Revise pages: " + (", ".join(map(str, weak_pages)) if weak_pages else "none, well done!"))

    # Save the whole session; Phase 6 uses these files to check grading accuracy.
    out = Path("data/sessions")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"notes": name, "model": LLM_MODEL, "total": total,
                                "questions": session}, indent=2), encoding="utf-8")
    print(f"Session saved to {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
    else:
        run_viva(sys.argv[1])