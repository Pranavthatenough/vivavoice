"""
VivaVoice - web server (Phase 4: now with voice).

Start it from the project folder:
    python backend/app.py
Then open http://127.0.0.1:8000 in your browser.
"""
import re
import threading

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import examiner
import voice
from examiner import LEVELS, PERSONAS, ExaminerUnavailable, VivaSession
from rag import ROOT, get_db, ingest

UPLOADS = ROOT / "data" / "uploads"
MAX_UPLOAD_MB = 25

app = FastAPI(title="VivaVoice")


def load_models():
    """
    Load the LLM first, while memory is emptiest, then Whisper.
    Doing it at startup means a memory problem shows up here, with a clear
    message, instead of halfway through a viva.
    """
    error = examiner.free_memory_and_load()
    if error:
        print(f"\nThe examiner's model ({examiner.LLM_MODEL}) could not load: {error}")
        print("Close other apps (especially browsers and games), then restart the server.\n")
    else:
        print(f"Examiner model ready ({examiner.LLM_MODEL}).")
    voice.warm_up()


@app.on_event("shutdown")
def on_shutdown():
    examiner.unload()
sessions = {}   # session id -> VivaSession (kept in memory while the server runs)


class StartRequest(BaseModel):
    notes: str
    persona: str = "internal"
    language: str = "en"


class AnswerRequest(BaseModel):
    answer: str


class SpeakRequest(BaseModel):
    text: str


def get_session(sid):
    if sid not in sessions:
        raise HTTPException(404, "This viva has ended or the server was restarted. Start a new one.")
    return sessions[sid]


def public_question(q):
    """What the student sees before answering: no key points, no evidence."""
    return {"number": q["number"], "question": q["question"], "page": q["page"],
            "level": LEVELS[q["level"]]["name"]}


def persona_public(key):
    p = PERSONAS[key]
    return {"key": key, "name": p["name"], "about": p["about"],
            "silence_ms": p["silence_ms"], "rate": p["rate"], "pitch": p["pitch"]}


@app.get("/api/personas")
def personas():
    return [persona_public(k) for k in PERSONAS]


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
        s = VivaSession(req.notes, req.persona, req.language)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(404, f"No notes called '{req.notes}'. Upload them first.")
    sessions[s.id] = s
    try:
        q = s.next_question()
    except ExaminerUnavailable as e:
        raise HTTPException(503, str(e))
    return {"session_id": s.id, "total": s.total, "persona": persona_public(s.persona_key),
            "language": s.language, "question": public_question(q)}


@app.post("/api/sessions/{sid}/answer")
def answer(sid: str, req: AnswerRequest):
    s = get_session(sid)
    try:
        r = s.answer(req.answer)
    except ValueError as e:
        raise HTTPException(409, str(e))
    except ExaminerUnavailable as e:
        # The answer isn't lost: the question stays open, so "Hand in answer" can be pressed again.
        raise HTTPException(503, str(e))
    if r["stage"] == "follow_up":
        return {"stage": "follow_up", "follow_up": r["follow_up"], "number": r["number"]}
    return {
        "stage": "final",
        "score": r["score"],
        "feedback": r["feedback"],
        "voice": r.get("voice"),
        "follow_up": ({"question": r["follow_up"]["question"], "answer": r["follow_up"].get("answer"),
                       "recovered": r["follow_up"].get("recovered")} if r.get("follow_up") else None),
        "key_points": [{"point": p["point"], "status": p["status"], "span": p["span"]}
                       for p in r["key_points"]],
        "page": r["page"],
        "level": LEVELS[r["level"]]["name"],
        "source_text": r["source_text"],
        "done": s.done,
    }


@app.post("/api/sessions/{sid}/next")
def next_question(sid: str):
    s = get_session(sid)
    try:
        q = s.next_question()
    except ExaminerUnavailable as e:
        raise HTTPException(503, str(e))
    if q is None:
        raise HTTPException(409, "No questions left. Open the report.")
    return public_question(q)


@app.post("/api/sessions/{sid}/transcribe")
def transcribe(sid: str, audio: UploadFile, waited: float = Form(0.0)):
    """
    Speech-to-text for the current question.
    `waited` = seconds between the question being ready and the microphone opening.
    """
    s = get_session(sid)
    q = s.current()
    if q is None:
        raise HTTPException(409, "There is no question waiting for an answer.")
    target = q["follow_up"] if q["stage"] == "follow_up" else q   # where the voice data is saved
    if not voice.stt_available():
        raise HTTPException(501, "Speech-to-text isn't installed. Run: pip install faster-whisper")
    data = audio.file.read()
    if len(data) < 1000:
        raise HTTPException(400, "The recording was empty. Check that the right microphone is selected.")
    try:
        t = voice.transcribe(data, context=s.prompt_text(), language=s.language)
    except Exception as e:
        raise HTTPException(500, f"Couldn't transcribe the recording: {e}")
    first = t["signals"]["first_word_at"]
    t["seconds_before_speaking"] = round(waited + first, 1) if first is not None else None
    target["voice"] = {k: v for k, v in t.items() if k != "words"}   # saved with the session
    return {"text": t["text"], "words": t["words"], "signals": t["signals"],
            "seconds_before_speaking": t["seconds_before_speaking"],
            "seconds_to_transcribe": t["seconds_to_transcribe"]}


@app.get("/api/voice")
def voice_status():
    return {"stt": voice.stt_available(), "stt_model": voice.WHISPER_MODEL, "tts": voice.tts_engine()}


@app.post("/api/speak")
def speak(req: SpeakRequest):
    """WAV audio from Piper. 404 tells the page to use the browser's own voice instead."""
    wav = voice.speak(req.text[:1000])
    if wav is None:
        raise HTTPException(404, "Piper isn't set up; the browser will speak instead.")
    return Response(wav, media_type="audio/wav")


@app.get("/api/sessions/{sid}/report")
def report(sid: str):
    return get_session(sid).report()


# Serve the web page (frontend folder) at the root address.
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")

if __name__ == "__main__":
    threading.Thread(target=load_models, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)