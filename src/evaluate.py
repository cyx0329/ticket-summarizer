"""Eval harness: compare base vs prompt-engineered vs LoRA fine-tuned
on a held-out set of ticket -> notification pairs.

Metrics:
  - ROUGE-1/2/L against the reference notification
  - faithfulness: fraction of summary content words grounded in the ticket
  - pii_leaks: regex hits for IPs, emails, internal hostnames, ticket IDs
  - compression: len(notification) / len(ticket)

Usage:
    python src/evaluate.py --systems base,prompt,lora --eval data/eval.jsonl
    python src/evaluate.py --systems base --eval data/eval.jsonl --limit 10  # smoke test
"""
import argparse
import json
import re
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = "outputs/adapter"
SYSTEM_BASIC = "Summarize the incident ticket below as a customer notification."
SYSTEM_ENGINEERED = (Path(__file__).resolve().parent.parent
                     / "prompts" / "summarize.txt").read_text().strip()

PII_PATTERNS = {
    "ipv4": re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "internal_host": re.compile(r"\b[\w-]+\.internal\.example\.com\b"),
    "ticket_id": re.compile(r"\[INC-\d+\]"),
}

STOPWORDS = set(
    "the a an and or of to in for on with is was were are be been by as at "
    "it its this that these those we you your our they them he she his her "
    "from has have had will would can could should may might do does did "
    "not no if then than so such only also about into over after before "
    "during".split()
)


def content_words(text: str) -> set[str]:
    return {w.strip(".,;:!?()\"'").lower() for w in text.split()
            if w.strip(".,;:!?()\"'").lower() not in STOPWORDS} - {""}


def faithfulness(summary: str, ticket: str) -> float:
    """Fraction of summary content words that appear in the source ticket.
    Crude, but catches the headline failure mode: invented specifics."""
    s_words = content_words(summary)
    if not s_words:
        return 0.0
    t_words = content_words(ticket)
    return sum(1 for w in s_words if w in t_words) / len(s_words)


def pii_leaks(summary: str) -> dict[str, int]:
    return {name: len(pat.findall(summary)) for name, pat in PII_PATTERNS.items()}


@torch.inference_mode()
def generate(model, tokenizer, system: str, ticket: str,
             max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": ticket},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def load_system(name: str):
    """Returns (model, tokenizer, system_prompt) for a named system."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if name == "lora":
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, ADAPTER_DIR).merge_and_unload()
        return model, tokenizer, SYSTEM_ENGINEERED
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    system = SYSTEM_ENGINEERED if name == "prompt" else SYSTEM_BASIC
    return model, tokenizer, system


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="base,prompt,lora")
    ap.add_argument("--eval", default="data/eval.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs/report.json")
    args = ap.parse_args()

    examples = [json.loads(l) for l in open(args.eval)]
    if args.limit:
        examples = examples[:args.limit]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                      use_stemmer=True)
    report = {}
    for name in args.systems.split(","):
        name = name.strip()
        print(f"evaluating system: {name}")
        model, tokenizer, system = load_system(name)
        rows, agg = [], {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0,
                         "faithful": 0.0, "pii": 0, "compression": 0.0}
        for ex in examples:
            pred = generate(model, tokenizer, system, ex["ticket"])
            scores = scorer.score(ex["notification"], pred)
            leaks = pii_leaks(pred)
            row = {
                "ticket": ex["ticket"][:200],
                "reference": ex["notification"],
                "prediction": pred,
                "rougeL": scores["rougeL"].fmeasure,
                "faithfulness": faithfulness(pred, ex["ticket"]),
                "pii_leaks": leaks,
                "compression": len(pred) / max(len(ex["ticket"]), 1),
            }
            rows.append(row)
            agg["rouge1"] += scores["rouge1"].fmeasure
            agg["rouge2"] += scores["rouge2"].fmeasure
            agg["rougeL"] += scores["rougeL"].fmeasure
            agg["faithful"] += row["faithfulness"]
            agg["pii"] += sum(leaks.values())
            agg["compression"] += row["compression"]
        n = len(rows)
        report[name] = {
            "n": n,
            "rouge1": round(agg["rouge1"] / n, 4),
            "rouge2": round(agg["rouge2"] / n, 4),
            "rougeL": round(agg["rougeL"] / n, 4),
            "faithfulness": round(agg["faithful"] / n, 4),
            "total_pii_leaks": agg["pii"],
            "avg_compression": round(agg["compression"] / n, 4),
            "examples": rows[:5],
        }
        del model
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {m: v[m] for m in
                           ("rougeL", "faithfulness", "total_pii_leaks",
                            "avg_compression")} for k, v in report.items()},
                     indent=2))
    print(f"full report -> {out}")


if __name__ == "__main__":
    main()
