"""Evaluate any OpenAI-compatible endpoint (vLLM, Azure OpenAI, Anthropic OpenAI-compat) on the golden set.

Example:
  python src/evaluate.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-0.5B-Instruct --name base
Set OPENAI_API_KEY for hosted endpoints (any value works for local vLLM).
"""
import argparse, json, os, time
from pathlib import Path
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument("--base-url", required=True)
p.add_argument("--model", required=True)
p.add_argument("--name", required=True, help="label for this run, e.g. base / lora / claude")
p.add_argument("--limit", type=int, default=200)
a = p.parse_args()

client = OpenAI(base_url=a.base_url, api_key=os.getenv("OPENAI_API_KEY", "local"))
labels = json.loads((ROOT / "data" / "labels.json").read_text())
rows = [json.loads(l) for l in open(ROOT / "data" / "golden_test.jsonl")][: a.limit]
system = "Reply with exactly one label from this list and nothing else:\n" + ", ".join(labels)

y_true, y_pred, lat = [], [], []
for r in rows:
    t0 = time.perf_counter()
    out = client.chat.completions.create(model=a.model, temperature=0, max_tokens=20,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": f"Message: {r['text']}\nLabel:"}])
    lat.append(time.perf_counter() - t0)
    pred = out.choices[0].message.content.strip().split()[0] if out.choices[0].message.content else ""
    y_true.append(r["label"]); y_pred.append(pred if pred in labels else "INVALID")

lat.sort()
result = {"run": a.name, "model": a.model, "n": len(rows),
          "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
          "accuracy": round(accuracy_score(y_true, y_pred), 4),
          "invalid_rate": round(y_pred.count("INVALID") / len(rows), 4),
          "p95_latency_s": round(lat[int(len(lat) * 0.95) - 1], 3)}
print(json.dumps(result, indent=2))
(ROOT / "outputs").mkdir(exist_ok=True)
with open(ROOT / "outputs" / "results.jsonl", "a") as f:
    f.write(json.dumps(result) + "\n")
