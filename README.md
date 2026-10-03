# ticket-summarizer

Fine-tune an open LLM to turn raw network-incident tickets into concise,
customer-facing notifications — with an eval harness that measures whether
the fine-tune was worth it.

This is the full MLE loop in one repo: **data → train (LoRA) → evaluate →
serve**. Built as a portfolio project for ML Engineer roles, in the same
problem domain as production incident-notification work.

## The setup

Three systems, one held-out eval set, scored the same way:

| System | What it is |
|---|---|
| `base` | Instruction-tuned 7B model, zero-shot with a basic prompt |
| `prompt` | Same model, carefully engineered prompt (formatting rules + examples) |
| `lora` | Same model + LoRA adapter fine-tuned on 400 ticket → notification pairs |

## Metrics that matter for this task

- **ROUGE-1/2/L** vs. reference notifications (content overlap)
- **Faithfulness proxy**: % of summary noun phrases/entities grounded in the source ticket (no invented hostnames, regions, or durations)
- **PII/sanitization check**: regex scan for leaked IPs, emails, internal hostnames, ticket IDs
- **Conciseness**: compression ratio (notification chars / ticket chars)

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate the dataset (synthetic network-incident tickets; replace with
#    your own CSV via --csv tickets.csv --text-col ticket --target-col notification)
python src/data_prep.py --train-size 400 --eval-size 100

# 2. Fine-tune (needs a GPU with ~16GB VRAM; 4-bit QLoRA)
python src/train.py --config configs/train.yaml

# 3. Evaluate all three systems
python src/evaluate.py --systems base,prompt,lora --eval data/eval.jsonl

# 4. Serve the winner (optional)
pip install -r requirements-serve.txt
python src/serve.py --model outputs/merged --port 8000
```

## Results

| System | ROUGE-L | Faithful | PII leaks | Avg. length |
|---|---|---|---|---|
| base | — | — | — | — |
| prompt | — | — | — | — |
| lora | — | — | — | — |

*(Fill in after running `evaluate.py` — the script writes `outputs/report.json`.)*

## Repo layout

```
configs/train.yaml      LoRA + training hyperparameters
src/data_prep.py        Synthetic ticket generator (or CSV ingester)
src/train.py            QLoRA supervised fine-tune with TRL
src/evaluate.py         Eval harness: base vs prompt vs LoRA
src/serve.py            vLLM OpenAI-compatible server + latency benchmark
```

## Roadmap

- [ ] LLM-as-judge faithfulness scoring (currently regex + overlap heuristics)
- [ ] DPO on preference pairs (good vs. bad notifications)
- [ ] RAG variant: retrieve similar past incidents as few-shot context
- [ ] Real dataset swap-in (internal tickets, properly anonymized)
