"""
VivaVoice - the examiner's memory of your past vivas.

Every finished viva is saved in data/sessions/. This module reads them back
to work out how well you know each page of your notes ("mastery"), so the
next viva can spend more questions on your weak pages.
"""
import json

from rag import ROOT

SESSIONS_DIR = ROOT / "data" / "sessions"
NEWEST_WEIGHT = 0.6     # how much the newest score counts against the older average


def past_sessions(notes):
    """All saved vivas for these notes, oldest first."""
    if not SESSIONS_DIR.exists():
        return []
    sessions = []
    for path in sorted(SESSIONS_DIR.glob(f"{notes}_*.json")):   # file names start with a timestamp
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("notes") == notes:        # "physics" must not match "physics_current"
            sessions.append(data)
    return sessions


def page_history(notes):
    """
    Mastery per page, from 0 to 5. Recent answers count more than old ones,
    so if you fixed a weak page, its mastery rises quickly.
    """
    sessions = past_sessions(notes)
    pages = {}
    for s in sessions:
        for q in s.get("questions", []):
            if "score" not in q or "page" not in q:
                continue
            p = pages.setdefault(q["page"], {"mastery": None, "asked": 0})
            m = p["mastery"]
            p["mastery"] = q["score"] if m is None else round(m + NEWEST_WEIGHT * (q["score"] - m), 2)
            p["asked"] += 1
    recent = {q.get("chunk_id") for q in sessions[-1].get("questions", [])} if sessions else set()
    return {"sessions": len(sessions), "pages": pages, "recent_chunks": recent - {None}}