"""QLoRA supervised fine-tune: ticket -> customer notification.

Usage:
    python src/train.py --config configs/train.yaml
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "summarize.txt"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_text_examples(train_file: str, tokenizer, system_prompt: str,
                        max_len: int) -> Dataset:
    rows = [json.loads(l) for l in open(train_file)]
    texts = []
    for r in rows:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": r["ticket"]},
            {"role": "assistant", "content": r["notification"]},
        ]
        texts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False))
    return Dataset.from_dict({"text": texts})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["model"]["bnb_4bit_compute_dtype"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["id"], quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ds = build_text_examples(cfg["data"]["train_file"], tokenizer,
                             system_prompt, cfg["training"]["max_seq_length"])

    peft_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type=cfg["lora"]["task_type"],
        bias="none",
    )
    t = cfg["training"]
    train_args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        gradient_checkpointing=t["gradient_checkpointing"],
        bf16=t["bf16"],
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        peft_config=peft_cfg,
        args=train_args,
        max_seq_length=t["max_seq_length"],
        dataset_text_field="text",
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(t["output_dir"])
    print(f"adapter saved to {t['output_dir']}")


if __name__ == "__main__":
    main()
