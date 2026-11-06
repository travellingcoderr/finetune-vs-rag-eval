"""Minimal OpenAI-compatible chat server on top of Hugging Face transformers (CPU, MPS or CUDA).

Use it when vLLM is not available (Mac laptops, CPU-only CI runners), or to serve a LoRA
adapter without merging it:

  python src/serve_local.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
  python src/serve_local.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/lora-banking77 --port 8001

Then point evaluate.py (or any OpenAI client) at http://localhost:8000/v1.
Greedy decoding, one request at a time. For throughput benchmarks use vLLM (llm-platform-k8s).
"""
from __future__ import annotations

import argparse
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--adapter", default=None, help="optional PEFT/LoRA adapter directory")
p.add_argument("--served-name", default=None, help="model name reported by the API")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
a = p.parse_args()

if a.threads:
    torch.set_num_threads(a.threads)
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32 if device == "cpu" else torch.bfloat16
tok = AutoTokenizer.from_pretrained(a.adapter or a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(device)
if a.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
model.eval()
NAME = a.served_name or (a.adapter or a.model)
lock = threading.Lock()
app = FastAPI()


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str | None = None
    messages: list[Msg]
    max_tokens: int = 256
    temperature: float = 0.0


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    prompt = tok.apply_chat_template([m.model_dump() for m in req.messages], tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(device)
    with lock, torch.inference_mode():
        t0 = time.perf_counter()
        out = model.generate(**ids, max_new_tokens=req.max_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        dt = time.perf_counter() - t0
    new = out[0, ids["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=True)
    finish = "length" if len(new) >= req.max_tokens and new[-1].item() != tok.eos_token_id else "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": ids["input_ids"].shape[1], "completion_tokens": len(new),
                  "total_tokens": ids["input_ids"].shape[1] + len(new)},
        "x_generation_s": round(dt, 3),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
