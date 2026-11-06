"""Evaluate an LLM on the golden set through any OpenAI-compatible endpoint.

Modes
  zero-shot  system prompt lists all 77 labels, model picks one
  rag        + the k most similar labeled training messages as few-shot examples (MiniLM retrieval)
  finetuned  the exact prompt the LoRA adapter was trained on (no label list needed)

Examples
  python src/evaluate.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-0.5B-Instruct --mode zero-shot --name qwen0.5b_zero_shot
  python src/evaluate.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-0.5B-Instruct --mode rag --name qwen0.5b_rag
  python src/evaluate.py --base-url http://localhost:8001/v1 --model lora --mode finetuned --name qwen0.5b_lora
  # frontier model (OpenAI-compatible API), cost from token prices:
  OPENAI_API_KEY=... python src/evaluate.py --base-url https://api.openai.com/v1 --model gpt-4.1-mini --mode zero-shot \
      --name gpt41mini_zero_shot --usd-per-1m-in 0.40 --usd-per-1m-out 1.60
"""
import argparse
import os
import json
import time

from openai import OpenAI

from common import (DATA, FT_PROMPT, OUTPUTS, append_result, cost_per_1k, load_labels, parse_label, percentile, rag_user,
                    read_jsonl, scores, zero_shot_system)

p = argparse.ArgumentParser()
p.add_argument("--base-url", required=True)
p.add_argument("--model", required=True)
p.add_argument("--name", required=True, help="label for this run in the results table")
p.add_argument("--mode", choices=["zero-shot", "rag", "finetuned"], default="zero-shot")
p.add_argument("--k", type=int, default=5, help="examples retrieved in rag mode")
p.add_argument("--limit", type=int, default=300)
p.add_argument("--usd-per-hour", type=float, default=None, help="self-hosted instance price")
p.add_argument("--usd-per-1m-in", type=float, default=None, help="API input token price")
p.add_argument("--usd-per-1m-out", type=float, default=None, help="API output token price")
p.add_argument("--hardware", default="", help="free text, e.g. '2 vCPU sandbox, transformers'")
a = p.parse_args()

client = OpenAI(base_url=a.base_url, api_key=os.getenv("OPENAI_API_KEY", "local"))
labels = load_labels()
rows = read_jsonl(DATA / "golden_test.jsonl", a.limit)

retrieve = None
if a.mode == "rag":
    from sentence_transformers import SentenceTransformer

    train = read_jsonl(DATA / "train.jsonl")
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    index = emb.encode([r["text"] for r in train], batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def retrieve(text):
        q = emb.encode([text], normalize_embeddings=True)[0]
        top = (index @ q).argsort()[::-1][: a.k]
        return [train[i] for i in top]


def messages_for(text):
    if a.mode == "finetuned":
        return [{"role": "user", "content": FT_PROMPT.format(text=text)}]
    if a.mode == "rag":
        return [{"role": "system", "content": zero_shot_system(labels)}, {"role": "user", "content": rag_user(text, retrieve(text))}]
    return [{"role": "system", "content": zero_shot_system(labels)}, {"role": "user", "content": f"Message: {text}\nLabel:"}]


y_true, y_pred, lat, tin, tout, raw = [], [], [], 0, 0, []
for i, r in enumerate(rows, 1):
    msgs = messages_for(r["text"])  # retrieval time counts toward latency below
    t0 = time.perf_counter()
    out = client.chat.completions.create(model=a.model, temperature=0, max_tokens=16, messages=msgs)
    lat.append(time.perf_counter() - t0)
    if out.usage:
        tin += out.usage.prompt_tokens
        tout += out.usage.completion_tokens
    y_true.append(r["label"])
    y_pred.append(parse_label(out.choices[0].message.content, labels))
    raw.append({"text": r["text"], "label": r["label"], "output": out.choices[0].message.content, "pred": y_pred[-1]})
    if i % 50 == 0:
        print(f"  {i}/{len(rows)} running acc={sum(t == p for t, p in zip(y_true, y_pred)) / i:.3f}", flush=True)

result = {"run": a.name, "approach": a.mode, "model": a.model, "n": len(rows), **scores(y_true, y_pred),
          "p50_latency_s": round(percentile(lat, 0.5), 3), "p95_latency_s": round(percentile(lat, 0.95), 3),
          "avg_prompt_tokens": round(tin / len(rows), 1), "avg_completion_tokens": round(tout / len(rows), 1),
          "usd_per_1k": cost_per_1k(lat, a.usd_per_hour, tin, tout, a.usd_per_1m_in, a.usd_per_1m_out),
          "hardware": a.hardware}
print(result)
append_result(result)
# raw outputs, so a run can be re-scored if parse_label changes (no model calls needed)
(OUTPUTS / f"preds-{a.name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in raw))
