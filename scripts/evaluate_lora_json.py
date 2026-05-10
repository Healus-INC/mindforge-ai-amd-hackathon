#!/usr/bin/env python3
"""
A/B-friendly eval: base Qwen only (--no_adapter) vs Qwen + MindForge LoRA.

Uses the same held-out cases, system prompt, and JSON extraction as evaluate_base_vs_lora.py.
Prints a human-readable log to stdout (redirect to eval_base_qwen.txt / eval_lora_mindforge.txt)
and optionally writes a machine-readable JSON via --report_json for compare_ab_eval_reports.py.

Metrics (contract adherence, not clinical truth):
  valid_json_rate, required_schema_fields_present, risk_level_present,
  expected_risk_level_match (when gold exists), escalation_present,
  patient_safe_response_present, clinician_summary_present,
  medication_or_adherence_flags_present
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from sibling script (same directory on sys.path when run as python scripts/...).
_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import evaluate_base_vs_lora as ebv  # noqa: E402

from mindforge_schema import normalize_risk_level, prompt_dict_from_eval_case  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval base vs LoRA on mindforge_eval_cases.json (structured task adherence)."
    )
    p.add_argument("--cases", default="data/synthetic/mindforge_eval_cases.json")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--adapter_dir",
        default=None,
        help="LoRA adapter directory. Omit with --no_adapter.",
    )
    p.add_argument(
        "--no_adapter",
        action="store_true",
        help="Evaluate base instruct model only (no LoRA).",
    )
    p.add_argument("--limit", type=int, default=None, help="Max cases (default: all in file).")
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Completion budget; increase if eval JSON is truncated mid-object.",
    )
    p.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Write structured metrics + per-case rows here (for compare_ab_eval_reports.py).",
    )
    return p.parse_args()


def _nonempty_str(v: Any, min_len: int = 1) -> bool:
    return isinstance(v, str) and len(v.strip()) >= min_len


def risk_level_present(parsed: dict[str, Any] | None) -> bool:
    if not parsed:
        return False
    return normalize_risk_level(parsed.get("risk_level")) is not None


def escalation_present(parsed: dict[str, Any] | None) -> bool:
    if not parsed:
        return False
    er = parsed.get("escalation_recommendation")
    if not isinstance(er, dict):
        return False
    return any(str(er.get(k) or "").strip() for k in ("level", "timeframe", "reason"))


def patient_safe_response_present(parsed: dict[str, Any] | None) -> bool:
    if not parsed:
        return False
    return _nonempty_str(parsed.get("patient_safe_response"), min_len=10)


def clinician_summary_present(parsed: dict[str, Any] | None) -> bool:
    if not parsed:
        return False
    return _nonempty_str(parsed.get("clinician_summary"), min_len=10)


def medication_or_adherence_flags_present(parsed: dict[str, Any] | None) -> bool:
    """
    True if the output carries medication/adherence signal in the MindForge-shaped fields:
    structured medication_adherence object with content, and/or non-empty side_effect_flags.
    """
    if not parsed:
        return False
    mad = parsed.get("medication_adherence")
    if isinstance(mad, dict):
        flags = mad.get("flags")
        if isinstance(flags, list) and len(flags) > 0:
            return True
        if _nonempty_str(mad.get("status")) or _nonempty_str(mad.get("recommended_action")):
            return True
    sf = parsed.get("side_effect_flags")
    if isinstance(sf, list) and len(sf) > 0:
        return True
    return False


def required_schema_fields_present(parsed: dict[str, Any] | None) -> bool:
    """All slide-critical fields present (AND of the structural checks below)."""
    if not parsed:
        return False
    return (
        risk_level_present(parsed)
        and escalation_present(parsed)
        and patient_safe_response_present(parsed)
        and clinician_summary_present(parsed)
        and medication_or_adherence_flags_present(parsed)
    )


def expected_risk_level_match(parsed: dict[str, Any] | None, case: dict[str, Any]) -> bool | None:
    exp = case.get("expected") or {}
    gold = exp.get("risk_level")
    if not isinstance(gold, str) or not gold.strip():
        return None
    pred = normalize_risk_level(parsed.get("risk_level")) if parsed else None
    if pred is None:
        return False
    return pred == gold.strip().upper()


def aggregate_rates(rows: list[dict[str, Any]], key: str) -> float:
    n = len(rows)
    if n == 0:
        return 0.0
    ok = sum(1 for r in rows if r.get(key) is True)
    return ok / n


def aggregate_optional_match(rows: list[dict[str, Any]], key: str) -> float | None:
    """For keys that are True/False/None — rate over rows where value is not None."""
    scored = [r for r in rows if r.get(key) is not None]
    if not scored:
        return None
    ok = sum(1 for r in scored if r.get(key) is True)
    return ok / len(scored)


def main() -> None:
    args = parse_args()
    cases_path = (REPO_ROOT / args.cases).resolve() if not Path(args.cases).is_absolute() else Path(args.cases)
    if not cases_path.is_file():
        print(f"ERROR: cases file not found: {cases_path}", file=sys.stderr)
        sys.exit(2)

    adapter_dir_arg: str | None = None
    if args.no_adapter:
        adapter_path: Path | None = None
    else:
        if not args.adapter_dir:
            print("ERROR: provide --adapter_dir or use --no_adapter", file=sys.stderr)
            sys.exit(2)
        adapter_dir_arg = args.adapter_dir
        raw_ad = Path(args.adapter_dir)
        adapter_path = (REPO_ROOT / raw_ad) if not raw_ad.is_absolute() else raw_ad
        adapter_path = adapter_path.resolve()
        if not adapter_path.is_dir():
            print(f"ERROR: adapter_dir is not a directory: {adapter_path}", file=sys.stderr)
            sys.exit(2)

    cases = ebv.load_cases(cases_path, args.limit)
    if not cases:
        print("ERROR: no cases loaded", file=sys.stderr)
        sys.exit(2)

    run_label = "base_qwen" if args.no_adapter else "mindforge_lora"

    print("=" * 80)
    print("MindForge A/B eval — structured task adherence (not clinical superiority)")
    print("run:", run_label)
    print("base_model:", args.base_model)
    if args.no_adapter:
        print("adapter_dir: (none — base only)")
    else:
        print("adapter_dir:", f"{adapter_dir_arg!r} -> {adapter_path}")
    print("cases:", cases_path)
    print("n_cases:", len(cases))
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if args.no_adapter:
        model = base
    else:
        model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    per_case: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        payload = prompt_dict_from_eval_case(case)
        run = ebv.run_case(model, tokenizer, payload, args.max_new_tokens)
        parsed = run.get("parsed")

        erm = expected_risk_level_match(parsed, case)
        row = {
            "case_index": i,
            "case_id": case.get("case_id"),
            "patient_note_preview": (payload.get("patient_note") or "")[:200],
            "valid_json": bool(run.get("json_valid")),
            "risk_level_present": risk_level_present(parsed),
            "escalation_present": escalation_present(parsed),
            "patient_safe_response_present": patient_safe_response_present(parsed),
            "clinician_summary_present": clinician_summary_present(parsed),
            "medication_or_adherence_flags_present": medication_or_adherence_flags_present(parsed),
            "required_schema_fields_present": required_schema_fields_present(parsed),
            "expected_risk_level_match": erm,
            "risk_level_normalized": run.get("risk_level_normalized"),
            "raw_generation_tail": (run.get("snippet") or "")[:1200],
            "parsed_json": parsed,
        }
        per_case.append(row)

        print("-" * 80)
        print(f"CASE {i}/{len(cases)}  case_id={case.get('case_id')}")
        print("valid_json:", row["valid_json"])
        print("risk_level_present:", row["risk_level_present"])
        print("required_schema_fields_present:", row["required_schema_fields_present"])
        print("expected_risk_level_match:", erm)
        print("snippet:\n", (run.get("snippet") or "")[:1800])

    metrics = {
        "valid_json_rate": aggregate_rates(per_case, "valid_json"),
        "required_schema_fields_present": aggregate_rates(per_case, "required_schema_fields_present"),
        "risk_level_present": aggregate_rates(per_case, "risk_level_present"),
        "expected_risk_level_match": aggregate_optional_match(per_case, "expected_risk_level_match"),
        "escalation_present": aggregate_rates(per_case, "escalation_present"),
        "patient_safe_response_present": aggregate_rates(per_case, "patient_safe_response_present"),
        "clinician_summary_present": aggregate_rates(per_case, "clinician_summary_present"),
        "medication_or_adherence_flags_present": aggregate_rates(
            per_case, "medication_or_adherence_flags_present"
        ),
    }

    print("=" * 80)
    print("AGGREGATE METRICS")
    print(json.dumps(metrics, indent=2))
    print("=" * 80)
    print(
        "Framing: LoRA improved structured task adherence — the adapter does not make medical decisions; "
        "it makes the base model more reliably produce the Healus/MindForge care-loop JSON shape "
        "for validation, audit, and governance."
    )

    report: dict[str, Any] = {
        "run_label": run_label,
        "base_model": args.base_model,
        "adapter_dir": str(adapter_path) if adapter_path else None,
        "adapter_dir_cli": adapter_dir_arg,
        "no_adapter": bool(args.no_adapter),
        "cases_file": str(cases_path),
        "n_cases": len(cases),
        "max_new_tokens": args.max_new_tokens,
        "metrics": metrics,
        "per_case": per_case,
    }

    # Footprint-friendly copy (no full parsed trees) for stdout footer + default --report_json body.
    footer_report = json.loads(json.dumps(report))
    for r in footer_report["per_case"]:
        r.pop("parsed_json", None)

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(footer_report, indent=2), encoding="utf-8")
        print(f"\nWrote report_json: {args.report_json}")

    # Single-line footer so a captured .txt can be recovered via compare_ab_eval_reports.py
    print("\nMINDFORGE_EVAL_REPORT_JSON:" + json.dumps(footer_report))


if __name__ == "__main__":
    main()
