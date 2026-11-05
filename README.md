# Project 02: Fine-Tune vs RAG vs Prompting, Measured

Pick one narrow classification task, then compare: (a) a base small model with prompting, (b) the same model with **RAG** (retrieve the 5 most similar labeled training examples as context), (c) the same model fine-tuned with **LoRA**, and (d) a frontier model (Claude / Azure OpenAI). Score all four with the same golden dataset and **macro-F1**, plus latency and cost.

**Why it matters:** senior AI roles ask "when do you fine-tune vs use RAG vs prompt?" This gives you a data-backed answer and fills the fine-tuning gap on your resume. It reuses your golden-dataset / macro-F1 evaluation experience from Eli Lilly.

## Suggested task (public data, no confidentiality issues)
Support-ticket intent classification, which mirrors your MaxAI helpdesk win. Options:
- `banking77` dataset on Hugging Face (77 banking intents)
- Or a smaller label set you define (10–15 intents)

## Milestones (about 2 weeks)
1. `src/prepare_data.py`: download the dataset, create train / golden-test JSONL in `data/`
2. `src/train_lora.py`: LoRA fine-tune `Qwen/Qwen2.5-0.5B-Instruct` (Mac M-series works for 0.5B; use a cloud GPU or Colab for bigger)
3. Serve the base and fine-tuned models with vLLM (Project 01) or locally
4. Add a RAG variant: embed `train.jsonl`, retrieve the top-5 similar examples per query, and pass them as few-shot context (extend `evaluate.py` with a `--rag` flag)
5. `src/evaluate.py`: run each model against the golden set → macro-F1, accuracy, p95 latency, cost
6. Write-up: results table + recommendation on when each approach wins

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/prepare_data.py
python src/train_lora.py --epochs 2
python src/evaluate.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-0.5B-Instruct --name base
```

## Resume bullet once finished
> Benchmarked LoRA fine-tuning vs prompting vs frontier LLMs on intent classification (golden dataset, macro-F1); the fine-tuned 0.5B model reached __ macro-F1 at __x lower cost and __ ms p95 latency.
