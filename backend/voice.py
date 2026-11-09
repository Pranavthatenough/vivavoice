"""
VivaVoice - Phase 4: the examiner's ears (speech-to-text) and voice (text-to-speech).

Speech-to-text: faster-whisper, running locally on the CPU.
Text-to-speech: Piper, if a voice file is in the voices/ folder. Otherwise the
browser speaks instead, using the voices built into Windows.
"""
import io
import re
import threading
import time
import wave
from functools import lru_cache

from rag import ROOT

WHISPER_MODEL = "small"        # "base" is faster but makes more mistakes
PIPER_VOICE = ROOT / "voices" / "en_US-lessac-medium.onnx"
PAUSE_SECONDS = 1.5            # silences at least this long count as pauses

# Whisper tends to clean up speech and drop "umm" and "uh". Starting the prompt
# with fillers nudges it to write down exactly what was said, so we can count them.
FILLER_PROMPT = "Umm, let me think. Uh, so, hmm, the answer is, basically, "
FILLERS = re.compile(r"\b(u+m+|u+h+|e+r+m*|h+m+|a+h+|basically|you know|i mean)\b", re.I)

_lock = threading.Lock()       # one transcription at a time keeps the laptop responsive


# ---------- Speech to text ----------

@lru_cache(maxsize=1)
def whisper():
    from faster_whisper import WhisperModel   # imported here so text mode works without it
    # int8 makes the model about 4x smaller and faster on a CPU, with little loss in accuracy.
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def stt_available():
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def warm_up():
    """Load the model in the background when the server starts (first load downloads it)."""
    if stt_available():
        try:
            whisper()
            print(f"Speech-to-text ready (whisper {WHISPER_MODEL}).")
        except Exception as e:
            print(f"Speech-to-text could not load: {e}")


def transcribe(audio: bytes, context: str = ""):
    """
    Turn a recorded answer into text, with a timestamp for every word.
    `context` is the question being answered. Putting it in Whisper's prompt
    makes subject words like "Wheatstone" or "drift velocity" come out right.
    Only the question goes in, never the key points: those would nudge Whisper
    towards "hearing" the right answer, and the grading would stop being fair.
    """
    t0 = time.time()
    prompt = FILLER_PROMPT + context[:600]
    with _lock:
        segments, info = whisper().transcribe(
            io.BytesIO(audio),
            language="en",
            initial_prompt=prompt,
            word_timestamps=True,
            vad_filter=True,                  # skip silence, stops Whisper inventing words in it
            condition_on_previous_text=False,
            beam_size=5,
        )
        segments = list(segments)             # the actual work happens here
    words = [{"word": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2)}
             for s in segments for w in (s.words or []) if w.word.strip()]
    text = " ".join(s.text.strip() for s in segments).strip()
    return {
        "text": text,
        "words": words,
        "audio_seconds": round(info.duration, 1),
        "seconds_to_transcribe": round(time.time() - t0, 1),
        "signals": speech_signals(text, words),
    }


def speech_signals(text, words):
    """How the answer was said: when speaking started, pauses, fillers and pace."""
    if not words:
        return {"first_word_at": None, "pauses": [], "fillers": [], "words_per_minute": None}
    pauses = [{"after": a["word"], "seconds": round(b["start"] - a["end"], 1)}
              for a, b in zip(words, words[1:]) if b["start"] - a["end"] >= PAUSE_SECONDS]
    fillers = [m.group(0).lower() for m in FILLERS.finditer(text)]
    speaking = words[-1]["end"] - words[0]["start"]
    wpm = round(len(words) / (speaking / 60)) if speaking >= 3 else None   # too short to judge pace
    return {"first_word_at": words[0]["start"], "pauses": pauses,
            "fillers": fillers, "words_per_minute": wpm}


# ---------- Text to speech ----------

@lru_cache(maxsize=1)
def piper_voice():
    if not PIPER_VOICE.exists():
        return None
    try:
        from piper import PiperVoice
    except ImportError:
        return None
    return PiperVoice.load(str(PIPER_VOICE))


def tts_engine():
    return "piper" if piper_voice() is not None else "browser"


def speak(text):
    """Return WAV audio of the text, or None if Piper isn't set up."""
    v = piper_voice()
    if v is None:
        return None
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        if hasattr(v, "synthesize_wav"):      # piper-tts 1.3 and newer
            v.synthesize_wav(text, wav)
        else:                                 # older piper-tts
            v.synthesize(text, wav)
    return buf.getvalue()