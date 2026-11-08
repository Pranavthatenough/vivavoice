"""
VivaVoice - Phase 3: web server.

Start it from the project folder:
    python backend/app.py
Then open http://127.0.0.1:8000 in your browser.
"""
import re

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from examiner import VivaSession
from rag import ROOT, get_db, ingest

UPLOADS = ROOT / "data" / "uploads"
MAX_UPLOAD_MB = 25

app = FastAPI(title="VivaVoice")
sessions = {}   # session id -> VivaSession (kept in memory while the server runs)


class StartRequest(BaseModel):
    notes: str


class AnswerRequest(BaseModel):
    answer: str


def get_session(sid):
    if sid not in sessions:
        raise HTTPException(404, "This viva has ended or the server was restarted. Start a new one.")
    return sessions[sid]


def public_question(q):
    """What the student sees before answering: no key points, no evidence."""
    return {"number": q["number"], "question": q["question"], "page": q["page"]}


@app.get("/api/notes")
def list_notes():
    db = get_db()
    return [{"name": c.name, "chunks": db.get_collection(c.name).count()}
            for c in db.list_collections()]


@app.post("/api/notes")
def upload_notes(file: UploadFile):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a PDF file.")
    data = file.file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"The file is larger than {MAX_UPLOAD_MB} MB.")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", file.filename)
    path = UPLOADS / safe
    path.write_bytes(data)
    try:
        name = ingest(str(path))
    except ValueError as e:              # e.g. a scanned PDF with no text
        raise HTTPException(400, str(e))
    return {"name": name}


@app.post("/api/sessions")
def start_session(req: StartRequest):
    try:
        s = VivaSession(req.notes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(404, f"No notes called '{req.notes}'. Upload them first.")
    sessions[s.id] = s
    return {"session_id": s.id, "total": s.total, "question": public_question(s.next_question())}


@app.post("/api/sessions/{sid}/answer")
def answer(sid: str, req: AnswerRequest):
    s = get_session(sid)
    try:
        r = s.answer(req.answer)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {
        "score": r["score"],
        "feedback": r["feedback"],
        "key_points": [{"point": p["point"], "covered": p["covered"], "span": p["span"]}
                       for p in r["key_points"]],
        "page": r["page"],
        "source_text": r["source_text"],
        "done": s.done,
    }


@app.post("/api/sessions/{sid}/next")
def next_question(sid: str):
    s = get_session(sid)
    q = s.next_question()
    if q is None:
        raise HTTPException(409, "No questions left. Open the report.")
    return public_question(q)


@app.get("/api/sessions/{sid}/report")
def report(sid: str):
    return get_session(sid).report()


# Serve the web page (frontend folder) at the root address.
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)