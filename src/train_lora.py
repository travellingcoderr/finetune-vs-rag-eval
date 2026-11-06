"""LoRA fine-tune a small instruct model to output the banking77 intent label.

Loss is computed only on the label tokens (prompt tokens are masked), and the prompt is built
with the model's chat template so the adapter can be served by any OpenAI-compatible server.

  python src/train_lora.py --epochs 2                 # data/train_lora.jsonl (10 per intent)
  python src/train_lora.py --data data/train.jsonl    # full split, use a GPU
"""
import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from common import DATA, FT_PROMPT, OUTPUTS, read_jsonl

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--data", default=str(DATA / "train_lora.jsonl"))
p.add_argument("--epochs", type=float, default=2)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--batch", type=int, default=2)
p.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch x grad-accum")
p.add_argument("--out", default=str(OUTPUTS / "lora-banking77"))
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
tok.pad_token = tok.pad_token or tok.eos_token
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32 if device == "cpu" else torch.bfloat16)
model.enable_input_require_grads()  # needed for gradient checkpointing with a frozen base
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                         task_type="CAUSAL_LM"))
model.print_trainable_parameters()

END = "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token


def encode(r):
    prompt = tok.apply_chat_template([{"role": "user", "content": FT_PROMPT.format(text=r["text"])}],
                                     tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(r["label"] + END, add_special_tokens=False)["input_ids"]
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


rows = [encode(r) for r in read_jsonl(Path(a.data))]


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda xs, v: [x + [v] * (n - len(x)) for x in xs]  # noqa: E731
    ids = torch.tensor(pad([b["input_ids"] for b in batch], tok.pad_token_id))
    labels = torch.tensor(pad([b["labels"] for b in batch], -100))
    return {"input_ids": ids, "labels": labels, "attention_mask": (labels != -100) | (ids != tok.pad_token_id)}


t0 = time.perf_counter()
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
                           gradient_checkpointing=True,
                           learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.05, logging_steps=10,
                           save_strategy="no", report_to=[], remove_unused_columns=False),
    train_dataset=rows,
    data_collator=collate,
)
out = trainer.train()
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
meta = {"base_model": a.model, "train_rows": len(rows), "epochs": a.epochs, "effective_batch": a.batch * a.grad_accum, "rank": a.rank, "lr": a.lr,
        "device": device, "train_s": round(time.perf_counter() - t0, 1), "final_loss": round(out.training_loss, 4)}
Path(a.out, "train_meta.json").write_text(json.dumps(meta, indent=2))
print("saved LoRA adapter ->", a.out, meta)
