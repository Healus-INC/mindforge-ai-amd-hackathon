"""Quick inference against the LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = """You are MindForge AI, a mental-health risk review assistant for a synthetic hackathon demo.
Return ONLY valid JSON. Do not diagnose, prescribe, or replace a clinician. Use conservative safety escalation.
Synthetic demo data only; no PHI.
All string values must be English only (US English); do not append text in Chinese or other languages."""

DEFAULT_CASE = {
    "case_id": "LIVE-DEMO-001",
    "context_type": "synthetic_demo",
    "patient_note": "Patient missed two doses this week, slept 3 hours last night, reports dizziness, and caregiver says the patient is increasingly withdrawn.",
    "device_or_app_events": {
        "doses_scheduled": 14,
        "doses_taken": 12,
        "missed_doses_7d": 2,
        "sleep_hours_last_night": 3,
        "mood_score_1_to_10": 3,
    },
    "task": "Return a structured mental-health risk review JSON for the care loop.",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--adapter_dir",
        default="outputs/mindforge-qwen-lora",
        help="Path to LoRA adapter (ignored when --base_only).",
    )
    p.add_argument(
        "--base_only",
        action="store_true",
        help="Run the base instruct model only (no LoRA). For before/after evals.",
    )
    p.add_argument("--case_json", default=None, help="Optional path to a JSON object case")
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Completion budget; raise if JSON is cut off before closing braces.",
    )
    return p.parse_args()


def extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


def main() -> None:
    args = parse_args()
    case = DEFAULT_CASE
    if args.case_json:
        case = json.loads(Path(args.case_json).read_text())

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if args.base_only:
        model = base
    else:
        model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(case)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    candidate = extract_json(generated)
    print(candidate)
    try:
        parsed = json.loads(candidate)
        print("\nJSON_VALID=true")
        print("risk_level=", parsed.get("risk_level"))
        print("risk_score=", parsed.get("risk_score"))
    except Exception as e:
        print("\nJSON_VALID=false", repr(e))


if __name__ == "__main__":
    main()
