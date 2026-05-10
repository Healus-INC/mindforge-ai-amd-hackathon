"""
MindForge AI LoRA SFT training script.

Default: Qwen/Qwen2.5-0.5B-Instruct for maximum hackathon speed.
Recommended command:
  python training/train_qwen_lora_trl.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --train_file data/synthetic/mindforge_train.cleaned.jsonl \
    --validation_file data/synthetic/mindforge_validation.cleaned.jsonl \
    --output_dir outputs/mindforge-qwen-lora \
    --epochs 6

The dataset must be JSONL with one object per line:
  {"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    p.add_argument(
        "--train_file",
        default="data/synthetic/mindforge_train.cleaned.jsonl",
        help="Use mindforge_train.cleaned.jsonl after prepare_sft_dataset.py",
    )
    p.add_argument(
        "--validation_file",
        default="data/synthetic/mindforge_validation.cleaned.jsonl",
        help="Use mindforge_validation.cleaned.jsonl after prepare_sft_dataset.py",
    )
    p.add_argument("--output_dir", default="outputs/mindforge-qwen-lora")
    p.add_argument("--epochs", type=float, default=6.0)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--save_steps", type=int, default=25)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_model_id", default=None)
    return p.parse_args()


def check_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing file: {path}")


def validate_jsonl(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            obj = json.loads(line)
            if "messages" not in obj:
                raise ValueError(f"{path}:{i} missing 'messages'")
            roles = [m.get("role") for m in obj["messages"]]
            if roles[-1] != "assistant":
                raise ValueError(f"{path}:{i} last message must be assistant, got {roles[-1]}")


def make_sft_config(args: argparse.Namespace) -> SFTConfig:
    common: dict[str, Any] = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        fp16=False,
        report_to="none",
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        packing=False,
        remove_unused_columns=False,
    )
    # TRL/Transformers naming changed across versions. Try compatible variants.
    for eval_key in ("eval_strategy", "evaluation_strategy"):
        for length_key in ("max_length", "max_seq_length"):
            try:
                return SFTConfig(**common, **{eval_key: "steps", length_key: args.max_length})
            except TypeError:
                pass
    # Last fallback with no max_length.
    for eval_key in ("eval_strategy", "evaluation_strategy"):
        try:
            return SFTConfig(**common, **{eval_key: "steps"})
        except TypeError:
            pass
    raise RuntimeError("Could not create SFTConfig; check TRL/Transformers versions.")


def main() -> None:
    args = parse_args()
    check_file(args.train_file)
    check_file(args.validation_file)
    validate_jsonl(args.train_file)
    validate_jsonl(args.validation_file)

    print("CUDA/HIP available:", torch.cuda.is_available())
    print("Torch HIP:", getattr(torch.version, "hip", None))
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: GPU not visible. Training will be slow or fail.")

    dataset = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.validation_file},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    sft_args = make_sft_config(args)
    trainer_kwargs = dict(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
    )

    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to: {args.output_dir}")

    if args.push_to_hub:
        trainer.push_to_hub()
        print(f"Pushed to Hub: {args.hub_model_id}")


if __name__ == "__main__":
    main()
