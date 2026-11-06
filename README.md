# Fine-tune vs RAG vs Prompting, measured

One narrow task, **Banking77 intent classification** (77 labels, public data), solved five ways and scored on the same golden slice with **macro-F1**, latency and cost: classical baselines, a small LLM zero-shot, the same LLM with retrieved examples (RAG), and the same LLM LoRA fine-tuned. A frontier-model row is ready to run once an API key is set.

## Results (real runs, November 2026)

n = 300 golden test messages, `Qwen/Qwen2.5-0.5B-Instruct`, temperature 0, served on a 2 vCPU / 8 GB sandbox with transformers (CPU, no GPU). Cost uses a c7i.large at $0.0892/hour ([Economize](https://www.economize.cloud/resources/aws/pricing/ec2/c7i.large/)) × busy time. Full table: [docs/results.md](docs/results.md), raw rows: `docs/results.jsonl`.

| Approach | Macro-F1 | Accuracy | Invalid label | p95 latency | $ / 1k predictions |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | **0.930** | 93.7% | 0% | 2 ms | ~$0 |
| MiniLM embeddings + 1-NN (the RAG retriever alone) | 0.924 | 94.0% | 0% | 0.48 s | $0.005 |
| MiniLM embeddings + logistic regression | 0.917 | 92.7% | 0% | 0.48 s | $0.006 |
| Qwen 0.5B + RAG (5 retrieved examples) | 0.825 | 83.7% | 3.7% | 2.65 s | $0.055 |
| Qwen 0.5B + LoRA (r=16, 770 rows, 1 epoch) | 0.492 | 49.3% | 19.0% | 0.95 s | $0.017 |
| Qwen 0.5B zero-shot (77 labels in the prompt) | 0.250 | 28.7% | 16.7% | 1.94 s | $0.041 |
| Frontier API model | _pending API key_ | | | | |

## What the numbers say

- **Beat the baseline first.** For a closed label set with 10k labeled examples, a TF-IDF classifier trained in 40 s beats every LLM variant here, at a fraction of the latency and cost. An LLM has to earn its place.
- **RAG is the cheapest big win for a small LLM:** retrieved examples took the same model from 0.25 to 0.82 macro-F1 with no training. It uses the full 10k training set at query time.
- **This LoRA run was data-starved:** 10 examples per intent and 1 epoch on CPU (18 min, final loss 1.24) more than halved the error of zero-shot and is 3x faster than RAG (60-token prompts instead of 540), but it still invents labels 19% of the time (`top_up_fee_charged`, `pin_not_working`). The obvious next run is the full 10k split for 2–3 epochs on a GPU, or constrained decoding to the label set.
- **Zero-shot fails mostly on format:** with 77 labels in the prompt, a 0.5B model makes up plausible names (`top_up_by_card`). Invalid outputs are counted as wrong, not dropped.

Caveats: one small model, one task, CPU latencies (a GPU cuts LLM latency by roughly an order of magnitude but not the ranking above), 300-message slice.

## Reproduce

```bash
pip install -r requirements.txt
python src/prepare_data.py                        # data/train.jsonl (10,003), train_lora.jsonl (770), golden_test.jsonl (500)
python src/baselines.py --limit 300

python src/serve_local.py --port 8000 &           # OpenAI-compatible server (or point at vLLM from llm-platform-k8s)
python src/evaluate.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-0.5B-Instruct \
  --mode zero-shot --name qwen0.5b_zero_shot --limit 300 --usd-per-hour 0.0892
python src/evaluate.py ... --mode rag --name qwen0.5b_rag_k5

python src/train_lora.py --epochs 1               # batch 2 x grad-accum 4, gradient checkpointing; fits in 8 GB RAM
python src/serve_local.py --adapter outputs/lora-banking77 --served-name qwen0.5b-lora --port 8001 &
python src/evaluate.py --base-url http://localhost:8001/v1 --model qwen0.5b-lora --mode finetuned --name qwen0.5b_lora_r16 --limit 300 --usd-per-hour 0.0892

# frontier row: any OpenAI-compatible API, priced per token
OPENAI_API_KEY=... python src/evaluate.py --base-url https://api.openai.com/v1 --model <model> \
  --mode zero-shot --name <model>_zero_shot --limit 300 --usd-per-1m-in <price> --usd-per-1m-out <price>

python src/report.py                              # docs/results.md
```

Every run also writes `outputs/preds-<run>.jsonl` with raw model outputs, so runs can be re-scored without calling the model again.

## Layout

| File | Purpose |
|---|---|
| `src/common.py` | prompts, tolerant label parser, macro-F1/accuracy/invalid rate, percentile, cost model |
| `src/baselines.py` | TF-IDF and embedding baselines |
| `src/evaluate.py` | zero-shot / RAG / fine-tuned evaluation against any OpenAI-compatible endpoint |
| `src/train_lora.py` | LoRA with loss on label tokens only, chat-template prompts |
| `src/serve_local.py` | minimal OpenAI-compatible server (base model or base + adapter) |
| `src/report.py` | results table |
| `tests/` | 14 unit tests (parser, metrics, cost, report); run in CI with only scikit-learn installed |
