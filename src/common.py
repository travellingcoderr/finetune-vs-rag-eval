"""Shared helpers: data loading, prompts, label parsing, metrics, cost."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# The fine-tuned model is trained on exactly this user turn, so evaluate it with the same one.
FT_PROMPT = "Classify the customer message into exactly one banking77 intent label.\nMessage: {text}\nLabel:"


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return rows[:limit] if limit else rows


def load_labels() -> list[str]:
    return json.loads((DATA / "labels.json").read_text())


def zero_shot_system(labels: list[str]) -> str:
    return ("You classify online-banking customer messages into one intent label.\n"
            "Reply with exactly one label from this list and nothing else:\n" + ", ".join(labels))


def rag_user(text: str, examples: list[dict]) -> str:
    shots = "\n".join(f"Message: {e['text']}\nLabel: {e['label']}" for e in examples)
    return f"Labeled examples of similar messages:\n{shots}\n\nNow classify this one.\nMessage: {text}\nLabel:"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def parse_label(output: str | None, labels: list[str]) -> str:
    """Map a raw model reply to a label, or 'INVALID'. Tolerates quotes, case, spaces vs underscores."""
    if not output:
        return "INVALID"
    first = output.strip().splitlines()[0] if output.strip() else ""
    first = re.sub(r"(?i)^(label|intent)\s*[:=-]\s*", "", first).strip(" `'\".")
    lookup = {_norm(l): l for l in labels}
    if _norm(first) in lookup:
        return lookup[_norm(first)]
    # Model added extra words: accept the longest label that appears as a whole token sequence.
    padded = f"_{_norm(output)}_"  # _norm turns every separator into "_", so match whole _-delimited tokens
    hits = [l for l in labels if f"_{_norm(l)}_" in padded]
    return max(hits, key=len) if hits else "INVALID"


def scores(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    labels = sorted(set(y_true))  # macro-F1 over classes present in the test set
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "invalid_rate": round(y_pred.count("INVALID") / max(1, len(y_pred)), 4),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = max(0, min(len(v) - 1, int(round(q * (len(v) - 1)))))
    return v[k]


def cost_per_1k(latencies: list[float], usd_per_hour: float | None, in_tok: int, out_tok: int,
                usd_per_1m_in: float | None, usd_per_1m_out: float | None) -> float | None:
    """Self-hosted: instance $/hour x busy time (sequential). API: token prices x token usage."""
    n = max(1, len(latencies))
    if usd_per_1m_in is not None or usd_per_1m_out is not None:
        total = in_tok / 1e6 * (usd_per_1m_in or 0) + out_tok / 1e6 * (usd_per_1m_out or 0)
        return round(total / n * 1000, 6)
    if usd_per_hour is not None:
        return round(sum(latencies) / 3600 * usd_per_hour / n * 1000, 6)
    return None


def append_result(result: dict) -> None:
    OUTPUTS.mkdir(exist_ok=True)
    with open(OUTPUTS / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
