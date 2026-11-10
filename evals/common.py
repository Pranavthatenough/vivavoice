"""
Shared helpers for the VivaVoice evaluations: file locations, crash-safe caching,
and the statistics every eval report uses.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

EVALS = Path(__file__).resolve().parent
ROOT = EVALS.parent
sys.path.insert(0, str(ROOT / "backend"))     # so evals can import the app's own code

GOLD = EVALS / "gold"                 # hand-written test data (committed)
RESULTS = EVALS / "results"           # eval outputs (committed: they are the evidence)
LABELS = EVALS / "labels.json"        # your human judgments from the review page (committed)
CACHE = ROOT / "data" / "eval_cache"  # work in progress (not committed)
SESSIONS = ROOT / "data" / "sessions"


def short_hash(*parts):
    """A short fingerprint. Used to tell prompt versions apart: change a prompt, get a new hash."""
    return hashlib.sha1("\n".join(map(str, parts)).encode("utf-8")).hexdigest()[:8]


class Cache:
    """
    Saves each result the moment it's ready, one line per result.
    If the laptop runs out of memory halfway, just run the eval again:
    finished items are skipped and it carries on where it stopped.
    """

    def __init__(self, name):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.path = CACHE / f"{name}.jsonl"
        self.data = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self.data[row["key"]] = row["value"]
                except (ValueError, KeyError):
                    pass                       # a half-written last line from a crash

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def save_result(name, data):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_result(name):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_labels():
    return json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}


def load_sessions():
    """Every saved viva, oldest first, with its file name."""
    out = []
    if SESSIONS.exists():
        for path in sorted(SESSIONS.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            data["_file"] = path.stem
            out.append(data)
    return out


# ---------- Statistics ----------

def pct(part, whole):
    return round(100 * part / whole, 1) if whole else None


def wilson(k, n, z=1.96):
    """
    95% confidence range for a percentage measured on a small sample.
    With 20 items, "90%" could really be anywhere from about 70% to 97%,
    and the report says so instead of pretending to be precise.
    """
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * max(0.0, centre - half), 1), round(100 * min(1.0, centre + half), 1)]


def binary_stats(pairs):
    """
    Compare true answers with predictions, e.g. (point really covered, grader said covered).
    Positive = "covered" / "yes".
    - false_positive_rate: how often it says yes when the truth is no. For grading this is
      "credit given for something the student never said", the most damaging mistake.
    - kappa: agreement beyond what chance alone would give (1 = perfect, 0 = no better than chance).
    """
    pairs = [(bool(t), bool(p)) for t, p in pairs]
    n = len(pairs)
    if not n:
        return None
    tp = sum(t and p for t, p in pairs)
    fp = sum(not t and p for t, p in pairs)
    fn = sum(t and not p for t, p in pairs)
    tn = n - tp - fp - fn
    agree = (tp + tn) / n
    chance = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    kappa = 1.0 if chance == 1 else (agree - chance) / (1 - chance)
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": pct(tp + tn, n), "accuracy_range": wilson(tp + tn, n),
        "precision": pct(tp, tp + fp), "recall": pct(tp, tp + fn),
        "false_positive_rate": pct(fp, fp + tn), "false_negative_rate": pct(fn, fn + tp),
        "kappa": round(kappa, 2),
    }


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def percentile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 2)


def progress(i, n, label):
    print(f"\r  {label}: {i}/{n}", end="" if i < n else "\n", flush=True)