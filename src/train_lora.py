"""LoRA fine-tune a small instruct model to output the intent label for a ticket."""
import argparse, json
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--epochs", type=float, default=2)
p.add_argument("--out", default=str(ROOT / "outputs" / "lora-banking77"))
a = p.parse_args()

labels = json.loads((ROOT / "data" / "labels.json").read_text())
PROMPT = ("Classify the customer message into exactly one intent label.\n"
          "Message: {text}\nLabel:")

tok = AutoTokenizer.from_pretrained(a.model)
tok.pad_token = tok.pad_token or tok.eos_token
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                         task_type="CAUSAL_LM"))
model.print_trainable_parameters()

ds = load_dataset("json", data_files=str(ROOT / "data" / "train.jsonl"))["train"]
def fmt(r):
    return tok(PROMPT.format(text=r["text"]) + " " + r["label"] + tok.eos_token, truncation=True, max_length=256)
ds = ds.map(fmt, remove_columns=ds.column_names)

Trainer(
    model=model,
    args=TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=8,
                           learning_rate=2e-4, logging_steps=20, save_strategy="epoch", report_to=[]),
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
).train()
model.save_pretrained(a.out); tok.save_pretrained(a.out)
print("saved LoRA adapter ->", a.out)
