"""
Eval 5: Speech-to-text. How accurately does Whisper write down what you said,
and do the Phase 4 tricks actually help?

You record yourself reading the 10 sentences in evals/gold/read_aloud.json once
(7 English, 3 Hinglish). Because the exact words are known, the error rate is
measured fairly. Then the same recordings are transcribed four ways:

  app          Whisper small + style prompt + the question    (what VivaVoice does)
  no_question  Whisper small + style prompt only              (does adding the question help?)
  no_prompt    Whisper small, no prompt at all                (do the prompt tricks help at all?)
  base         Whisper base  + style prompt + the question    (the smaller, faster model)

Measured: word error rate (English), subject-term accuracy (did "Wheatstone" come
out right?), filler words caught, and speed.

Recording: use Windows Sound Recorder (or any recorder), save each sentence as
evals/voice/<id>.m4a, e.g. evals/voice/e1.m4a. .wav, .mp3 and .webm also work.

Run from the project folder (Whisper only, no LLM; about 2-4 minutes):
    python evals/voice_eval.py
"""
import os

# OpenBLAS (used by numpy, which Whisper's audio code loads) reserves memory for every
# CPU thread at startup, and fails with "Memory allocation still failed" when RAM is low.
# One thread is plenty here; it must be set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import gc
import json
import re
import time

from common import EVALS, GOLD, ROOT, mean, pct, progress, save_result
import voice

VOICE_DIR = EVALS / "voice"
AUDIO_TYPES = (".m4a", ".wav", ".mp3", ".webm", ".ogg")
CONFIGS = [
    {"name": "app", "model": voice.WHISPER_MODEL, "style": True, "question": True},
    {"name": "no_question", "model": voice.WHISPER_MODEL, "style": True, "question": False},
    {"name": "no_prompt", "model": voice.WHISPER_MODEL, "style": False, "question": False},
    {"name": "base", "model": "base", "style": True, "question": True},
]
DIGITS = "zero one two three four five six seven eight nine".split()


def load_model(size):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


def normalise(text):
    """Lowercase, no punctuation, digits spelled out ("3.6" -> "three point six")."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"(\d)\.(\d)", r"\1 point \2", text)
    text = re.sub(r"\d", lambda m: f" {DIGITS[int(m.group())]} ", text)
    return re.findall(r"[a-z']+|[ऀ-ॿ]+", text)


def word_errors(reference, hypothesis):
    """Word-level edit distance: substitutions + deletions + insertions."""
    a, b = normalise(reference), normalise(hypothesis)
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i]
        for j, wb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb)))
        prev = cur
    return prev[-1], len(a)


def term_found(term, hypothesis):
    return f" {' '.join(normalise(term))} " in f" {' '.join(normalise(hypothesis))} "


def free_ollama_memory():
    """The LLM evals leave the model loaded in Ollama for 30 minutes. Whisper needs that memory."""
    try:
        import ollama
        for m in ollama.ps().models:
            name = getattr(m, "model", None) or getattr(m, "name", "")
            ollama.generate(model=name, keep_alive=0)
            print(f"Unloaded {name} from Ollama to free memory for Whisper.")
    except Exception:
        pass                                  # Ollama not running: nothing to free


def main():
    free_ollama_memory()
    gold = json.loads((GOLD / "read_aloud.json").read_text(encoding="utf-8"))["sentences"]
    recordings = {}
    for s in gold:
        for ext in AUDIO_TYPES:
            if (VOICE_DIR / f"{s['id']}{ext}").exists():
                recordings[s["id"]] = VOICE_DIR / f"{s['id']}{ext}"
                break
    missing = [s["id"] for s in gold if s["id"] not in recordings]
    if not recordings:
        print(f"No recordings found. Record the sentences in evals/gold/read_aloud.json and save them as "
              f"{VOICE_DIR.relative_to(ROOT)}\\e1.m4a, e2.m4a ... h3.m4a")
        return
    if missing:
        print(f"Missing recordings (skipped): {', '.join(missing)}")
    sentences = [s for s in gold if s["id"] in recordings]

    results = {}
    for ci, cfg in enumerate(CONFIGS, 1):
        print(f"[{ci}/{len(CONFIGS)}] {cfg['name']}: loading whisper {cfg['model']}")
        model = load_model(cfg["model"])
        rows = []
        for i, s in enumerate(sentences, 1):
            audio = recordings[s["id"]].read_bytes()
            if cfg["style"]:
                prompt = voice.build_prompt(s["context"] if cfg["question"] else "", s["language"])
            else:
                prompt = None
            t0 = time.time()
            segments, info = voice.run_whisper(model, audio, prompt, s["language"])
            seconds = time.time() - t0
            text = " ".join(seg.text.strip() for seg in segments).strip()
            errors, words = word_errors(s["text"], text)
            rows.append({
                "id": s["id"], "language": s["language"], "heard": text,
                "errors": errors, "words": words,
                "terms": len(s["terms"]), "terms_found": sum(term_found(t, text) for t in s["terms"]),
                "fillers_expected": s.get("fillers", 0),
                "fillers_found": len(voice.FILLERS.findall(text)),
                "seconds": round(seconds, 2), "audio_seconds": round(info.duration, 2),
            })
            progress(i, len(sentences), "  sentences")
        del model
        gc.collect()                      # free the model before loading the next one

        en = [r for r in rows if r["language"] == "en"]
        hi = [r for r in rows if r["language"] == "hinglish"]
        results[cfg["name"]] = {
            "config": cfg,
            "english_wer": pct(sum(r["errors"] for r in en), sum(r["words"] for r in en)),
            "english_terms": pct(sum(r["terms_found"] for r in en), sum(r["terms"] for r in en)),
            # Whisper writes Hindi in Devanagari, so a word error rate against the Roman-script
            # reference would be meaningless. For Hinglish, what matters is the English terms.
            "hinglish_terms": pct(sum(r["terms_found"] for r in hi), sum(r["terms"] for r in hi)),
            "fillers_caught": pct(sum(min(r["fillers_found"], r["fillers_expected"]) for r in rows),
                                  sum(r["fillers_expected"] for r in rows)),
            "seconds_per_sentence": mean([r["seconds"] for r in rows]),
            "speed_vs_realtime": round(sum(r["seconds"] for r in rows) /
                                       max(sum(r["audio_seconds"] for r in rows), 0.1), 2),
            "sentences": rows,
        }

    print(f"\n{'setup':12} {'EN WER':>7} {'EN terms':>9} {'HI terms':>9} {'fillers':>8} {'s/sent':>7} {'x realtime':>10}")
    for name, r in results.items():
        cells = [r["english_wer"], r["english_terms"], r["hinglish_terms"], r["fillers_caught"]]
        cells = ["-" if c is None else f"{c}%" for c in cells]
        print(f"{name:12} {cells[0]:>7} {cells[1]:>9} {cells[2]:>9} {cells[3]:>8} "
              f"{r['seconds_per_sentence']:>7} {r['speed_vs_realtime']:>10}")
    app = results["app"]
    worst = sorted([r for r in app["sentences"] if r["language"] == "en" and r["errors"]],
                   key=lambda r: -r["errors"])[:3]
    if worst:
        print("\nHardest English sentences for the app setup:")
    for r in worst:
        ref = next(s["text"] for s in sentences if s["id"] == r["id"])
        print(f"  {r['id']}: said  \"{ref}\"\n      heard \"{r['heard']}\"")

    path = save_result("voice", {"recordings": len(sentences), "missing": missing, "by_setup": results})
    print(f"\nSaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()