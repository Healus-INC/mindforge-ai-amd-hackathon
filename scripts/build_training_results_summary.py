#!/usr/bin/env python3
"""
Merge dataset stats + optional base-vs-LoRA eval reports into one JSON for the Gradio
Training Results tab and for presentation handouts.

Run after overnight training and (optionally) after:
  python scripts/evaluate_base_vs_lora.py --base_model Qwen/Qwen2.5-1.5B-Instruct \\
    --adapter_dir outputs/mindforge-qwen-lora-scale \\
    --out_json outputs/base_vs_lora_scale_report.json

Then:
  python scripts/build_training_results_summary.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent


def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def slim_eval_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep only rates and counts needed for charts (drop per_case)."""
    jp = report.get("json_parse") or {}
    ts = report.get("training_schema_contract_full") or {}
    core = report.get("schema_core_relaxed") or {}
    rm = report.get("risk_level_exact_match_vs_gold") or {}
    es = report.get("escalation_match_vs_gold") or {}
    base_j = jp.get("base") or {}
    lora_j = jp.get("lora") or {}
    base_t = ts.get("base") or {}
    lora_t = ts.get("lora") or {}
    base_c = core.get("base") or {}
    lora_c = core.get("lora") or {}
    return {
        "n_cases": base_j.get("total"),
        "json_parse": {
            "base_pass": base_j.get("pass"),
            "lora_pass": lora_j.get("pass"),
            "base_rate": base_j.get("rate"),
            "lora_rate": lora_j.get("rate"),
        },
        "training_schema_full": {
            "base_pass": base_t.get("pass"),
            "lora_pass": lora_t.get("pass"),
            "base_rate": base_t.get("rate"),
            "lora_rate": lora_t.get("rate"),
        },
        "schema_core": {
            "base_pass": base_c.get("pass"),
            "lora_pass": lora_c.get("pass"),
            "base_rate": base_c.get("rate"),
            "lora_rate": lora_c.get("rate"),
        },
        "risk_level_exact_match": {
            "n_gold": rm.get("total_with_gold"),
            "base_correct": rm.get("base_correct"),
            "lora_correct": rm.get("lora_correct"),
            "base_rate": rm.get("base_rate"),
            "lora_rate": rm.get("lora_rate"),
        },
        "escalation_match": {
            "n_gold": es.get("total_with_gold"),
            "base_correct": es.get("base_correct"),
            "lora_correct": es.get("lora_correct"),
            "base_rate": es.get("base_rate"),
            "lora_rate": es.get("lora_rate"),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build outputs/training_results_summary.json")
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs/training_results_summary.json",
        help="Output JSON path",
    )
    p.add_argument(
        "--scale-eval",
        type=Path,
        default=REPO_ROOT / "outputs/base_vs_lora_scale_report.json",
        help="evaluate_base_vs_lora output for scale adapter (optional)",
    )
    p.add_argument(
        "--demo-eval",
        type=Path,
        default=REPO_ROOT / "outputs/base_vs_lora_report.json",
        help="evaluate_base_vs_lora output for demo adapter (optional)",
    )
    p.add_argument(
        "--prepare-summary",
        type=Path,
        default=REPO_ROOT / "data/synthetic/prepare_sft_dataset_summary.json",
        help="Last prepare_sft_dataset summary (scale run)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prepare = load_json(args.prepare_summary)

    scale_train = REPO_ROOT / "data/synthetic/mindforge_train.scale.cleaned.jsonl"
    scale_val = REPO_ROOT / "data/synthetic/mindforge_validation.scale.cleaned.jsonl"
    demo_train = REPO_ROOT / "data/synthetic/mindforge_train.cleaned.jsonl"
    demo_val = REPO_ROOT / "data/synthetic/mindforge_validation.cleaned.jsonl"

    scale_dataset = {
        "train_file": "data/synthetic/mindforge_train.scale.cleaned.jsonl",
        "validation_file": "data/synthetic/mindforge_validation.scale.cleaned.jsonl",
        "train_rows": count_jsonl_lines(scale_train),
        "validation_rows": count_jsonl_lines(scale_val),
        "risk_distribution_train": (prepare or {}).get("train_risk_distribution"),
        "risk_distribution_validation": (prepare or {}).get("validation_risk_distribution"),
        "notes": (
            "100× package rows were normalized to the canonical 13-key SFT assistant JSON "
            "(scripts/normalize_100x_assistant_to_sft.py) before prepare_sft_dataset.py."
        ),
    }

    demo_dataset = {
        "train_file": "data/synthetic/mindforge_train.cleaned.jsonl",
        "validation_file": "data/synthetic/mindforge_validation.cleaned.jsonl",
        "train_rows": count_jsonl_lines(demo_train),
        "validation_rows": count_jsonl_lines(demo_val),
    }

    scale_report = load_json(args.scale_eval)
    demo_report = load_json(args.demo_eval)

    eval_block: dict[str, Any] = {
        "scale": None,
        "demo": None,
        "primary": None,
    }
    if isinstance(scale_report, dict):
        eval_block["scale"] = {
            "label": "Scale / overnight adapter (100× SFT)",
            "base_model": scale_report.get("base_model"),
            "adapter_dir": scale_report.get("adapter_dir"),
            "cases_file": scale_report.get("cases_file"),
            "metrics": slim_eval_report(scale_report),
        }
    if isinstance(demo_report, dict):
        eval_block["demo"] = {
            "label": "Demo adapter (smaller SFT run)",
            "base_model": demo_report.get("base_model"),
            "adapter_dir": demo_report.get("adapter_dir"),
            "cases_file": demo_report.get("cases_file"),
            "metrics": slim_eval_report(demo_report),
        }
    # Prefer scale metrics for the main chart when both exist (matches "new training set").
    if eval_block["scale"]:
        eval_block["primary"] = "scale"
    elif eval_block["demo"]:
        eval_block["primary"] = "demo"

    doc: dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_objective": (
            "Supervised fine-tuning (LoRA SFT) on structured chat JSONL so Qwen2.5-Instruct "
            "outputs a single MindForge assistant object: risk_level, risk_score, primary concerns, "
            "medication adherence, side-effect / sleep-mood / safety flags, escalation_recommendation, "
            "patient_safe_response, clinician_summary, care_loop_targets, and missing_info. "
            "Training uses the full SFT contract in code (including a boolean disclaimer field). "
            "No RL / GRPO on top."
        ),
        "methodology": [
            "Parameter-efficient LoRA (PEFT) on Qwen attention + MLP projections.",
            "TRL SFTTrainer, causal LM, bf16 on GPU, cosine LR schedule.",
            "Overnight scale defaults: Qwen/Qwen2.5-1.5B-Instruct, 3× epochs, "
            "data/synthetic/*.scale.cleaned.jsonl → outputs/mindforge-qwen-lora-scale.",
            "Demo path: Qwen/Qwen2.5-0.5B-Instruct → outputs/mindforge-qwen-lora.",
        ],
        "scale_dataset": scale_dataset,
        "demo_dataset": demo_dataset,
        "overnight_run": {
            "script": "training/run_overnight_scale.sh",
            "default_base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "default_output_dir": "outputs/mindforge-qwen-lora-scale",
            "default_epochs": 3,
            "default_batch_grad_accum": "per_device_batch=1, grad_accum=8",
            "default_learning_rate": 2e-4,
            "default_max_length": 2048,
        },
        "eval": eval_block,
        "how_to_refresh_eval": (
            "cd repo && source .venv/bin/activate\n"
            "# Scale adapter (must match BASE_MODEL used at train time):\n"
            "python scripts/evaluate_base_vs_lora.py \\\n"
            "  --base_model Qwen/Qwen2.5-1.5B-Instruct \\\n"
            "  --adapter_dir outputs/mindforge-qwen-lora-scale \\\n"
            "  --out_json outputs/base_vs_lora_scale_report.json \\\n"
            "  --out_md outputs/base_vs_lora_scale_report.md\n"
            "# Then rebuild this summary:\n"
            "python scripts/build_training_results_summary.py"
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"  primary eval chart: {eval_block['primary'] or 'none (add eval JSON first)'}")


if __name__ == "__main__":
    main()
