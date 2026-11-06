"""Download banking77 and write the splits used by every experiment.

data/train.jsonl        full training split (10,003 rows): retrieval index for RAG + classical baselines
data/train_lora.jsonl   balanced subset for LoRA (default 10 per intent = 770 rows, small enough for a laptop CPU)
data/golden_test.jsonl  fixed, shuffled sample of the official test split (seed 42), never used for training
data/labels.json        the 77 intent names
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data"
p = argparse.ArgumentParser()
p.add_argument("--per-class", type=int, default=10, help="LoRA training examples per intent")
p.add_argument("--golden", type=int, default=500, help="golden test rows to keep")
a = p.parse_args()

OUT.mkdir(exist_ok=True)
rng = random.Random(42)
ds = load_dataset("PolyAI/banking77")
labels = ds["train"].features["label"].names


def rows(split):
    return [{"text": r["text"], "label": labels[r["label"]]} for r in ds[split]]


def dump(rs, name):
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rs):>6} rows -> data/{name}")


train = rows("train")
rng.shuffle(train)
dump(train, "train.jsonl")

by_label = defaultdict(list)
for r in train:
    by_label[r["label"]].append(r)
lora = [r for l in labels for r in by_label[l][: a.per_class]]
rng.shuffle(lora)
dump(lora, "train_lora.jsonl")

test = rows("test")
rng.shuffle(test)
dump(test[: a.golden], "golden_test.jsonl")
(OUT / "labels.json").write_text(json.dumps(labels, indent=2))
print(f"{len(labels)} labels -> data/labels.json")
