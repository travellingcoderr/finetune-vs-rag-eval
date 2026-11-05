"""Download banking77 and write train / golden-test JSONL files to data/."""
import json, random
from pathlib import Path
from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
random.seed(42)

ds = load_dataset("PolyAI/banking77")
labels = ds["train"].features["label"].names

def dump(split, path, limit=None):
    rows = list(ds[split])
    random.shuffle(rows)
    rows = rows[:limit] if limit else rows
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "label": labels[r["label"]]}) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")

dump("train", OUT / "train.jsonl", limit=3000)
dump("test", OUT / "golden_test.jsonl", limit=500)
(OUT / "labels.json").write_text(json.dumps(labels, indent=2))
