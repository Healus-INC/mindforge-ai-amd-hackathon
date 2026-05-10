#!/usr/bin/env python3
"""
Before / after: same eval cases with base-only vs base+LoRA.

Metrics:
  - JSON parse success (balanced extraction)
  - Training assistant schema contract (same keys/types as validate_sft_jsonl.py)
  - Gold risk_level / escalation from eval case `expected` (prompt excludes expected — no label leak)

Loads the base model once, runs all cases, attaches LoRA with PeftModel, runs again.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindforge_schema import (
    escalation_match,
    gold_escalation_slug,
    normalize_risk_level,
    prompt_dict_from_eval_case,
    schema_core_ok,
    training_contract_ok,
)

# Match infer_lora.py defaults for generation behavior.
SYSTEM = """You are MindForge AI, a mental-health risk review assistant for a synthetic hackathon demo.
Return ONLY one JSON object (no markdown fences). Do not diagnose, prescribe, or replace a clinician.
Use conservative safety escalation. Synthetic demo data only; no PHI.

Language: All string values must be English only (US English). Do not use Chinese or other scripts.

Required shape (match keys and types):
- risk_level: one of LOW, MODERATE, HIGH, CRISIS (uppercase)
- risk_score: integer 0-100
- not_medical_advice: true (boolean)
- escalation_recommendation: object with string fields level, timeframe, reason
- primary_concerns, medication_adherence, side_effect_flags, sleep_mood_flags, safety_flags,
  patient_safe_response, clinician_summary, care_loop_targets, missing_info
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare base vs LoRA on parse, schema, and gold labels.")
    p.add_argument("--cases", default="data/synthetic/mindforge_eval_cases.json")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter_dir", default="outputs/mindforge-qwen-lora")
    p.add_argument("--limit", type=int, default=None, help="Max cases (default: all)")
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Completion budget; increase if eval JSON is truncated mid-object.",
    )
    p.add_argument(
        "--out_json",
        default="outputs/base_vs_lora_report.json",
        help="Write structured results here.",
    )
    p.add_argument(
        "--out_md",
        default="outputs/base_vs_lora_report.md",
        help="Write human-readable summary here.",
    )
    return p.parse_args()


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if "```" not in t:
        return t
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, re.IGNORECASE)
    return m.group(1).strip() if m else t


def first_balanced_json_object(text: str) -> str | None:
    """First JSON object with correct `{`/`}` depth (strings ignored)."""
    for start in (m.start() for m in re.finditer(r"\{", text)):
        depth = 0
        in_string = False
        escape = False
        for j in range(start, len(text)):
            ch = text[j]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : j + 1]
    return None


def extract_json(text: str) -> str:
    cleaned = _strip_code_fences(text)
    balanced = first_balanced_json_object(cleaned)
    if balanced is not None:
        return balanced
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return cleaned
    return cleaned[start : end + 1]


def load_cases(path: Path, limit: int | None) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        cases = raw.get("cases", raw.get("items", []))
    else:
        cases = raw
    if limit is not None:
        cases = cases[:limit]
    return cases


def run_case(model, tokenizer, user_payload: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(user_payload)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    candidate = extract_json(generated)
    parsed: dict[str, Any] | None = None
    err = None
    try:
        parsed = json.loads(candidate)
    except Exception as e:
        err = repr(e)
    schema_strict, schema_strict_reasons = training_contract_ok(parsed)
    schema_core, schema_core_reasons = schema_core_ok(parsed)
    norm_risk = normalize_risk_level(parsed.get("risk_level")) if parsed else None
    return {
        "json_valid": parsed is not None,
        "schema_contract_ok": schema_strict,
        "schema_contract_reasons": schema_strict_reasons,
        "schema_core_ok": schema_core,
        "schema_core_reasons": schema_core_reasons,
        "risk_level_raw": parsed.get("risk_level") if parsed else None,
        "risk_level_normalized": norm_risk,
        "snippet": candidate[:400] + ("..." if len(candidate) > 400 else ""),
        "parse_error": err,
        "parsed": parsed,
    }


def summarize_bool(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    n = len(rows)
    ok = sum(1 for r in rows if r.get(key))
    return {"total": n, "pass": ok, "rate": ok / max(n, 1)}


def gold_risk(case: dict[str, Any]) -> str | None:
    exp = case.get("expected") or {}
    rl = exp.get("risk_level")
    if isinstance(rl, str) and rl.strip():
        return rl.strip().upper()
    return None


def risk_match(pred_norm: str | None, gold: str | None) -> bool:
    if gold is None or pred_norm is None:
        return False
    return pred_norm == gold


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"Missing cases file: {cases_path}", file=sys.stderr)
        sys.exit(2)

    cases = load_cases(cases_path, args.limit)
    if not cases:
        print("No cases to evaluate.", file=sys.stderr)
        sys.exit(2)

    print("Loading tokenizer + base model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    base.eval()

    base_rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        payload = prompt_dict_from_eval_case(case)
        print(f"  [BASE] case {i}/{len(cases)}", flush=True)
        r = run_case(base, tokenizer, payload, args.max_new_tokens)
        base_rows.append({"case_index": i, "case_id": case.get("case_id"), **r})

    adapter_path = Path(args.adapter_dir)
    if not adapter_path.is_dir():
        print(f"Missing adapter dir: {adapter_path}", file=sys.stderr)
        sys.exit(2)

    print("Loading LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    lora_rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        payload = prompt_dict_from_eval_case(case)
        print(f"  [LoRA] case {i}/{len(cases)}", flush=True)
        r = run_case(model, tokenizer, payload, args.max_new_tokens)
        lora_rows.append({"case_index": i, "case_id": case.get("case_id"), **r})

    base_parse = summarize_bool(base_rows, "json_valid")
    lora_parse = summarize_bool(lora_rows, "json_valid")
    base_schema = summarize_bool(base_rows, "schema_contract_ok")
    lora_schema = summarize_bool(lora_rows, "schema_contract_ok")
    base_core = summarize_bool(base_rows, "schema_core_ok")
    lora_core = summarize_bool(lora_rows, "schema_core_ok")

    gold_risk_cases = [c for c in cases if gold_risk(c)]
    n_gold_risk = len(gold_risk_cases)
    base_risk_correct = 0
    lora_risk_correct = 0
    base_esc_correct = 0
    lora_esc_correct = 0
    n_gold_esc = sum(1 for c in cases if gold_escalation_slug(c))

    per_case: list[dict[str, Any]] = []
    for case, br, lr in zip(cases, base_rows, lora_rows, strict=True):
        g = gold_risk(case)
        rm_b = risk_match(br.get("risk_level_normalized"), g)
        rm_l = risk_match(lr.get("risk_level_normalized"), g)
        if g:
            base_risk_correct += int(rm_b)
            lora_risk_correct += int(rm_l)

        esc_b = escalation_match(br.get("parsed"), case)
        esc_l = escalation_match(lr.get("parsed"), case)
        if gold_escalation_slug(case):
            if esc_b is True:
                base_esc_correct += 1
            if esc_l is True:
                lora_esc_correct += 1

        per_case.append(
            {
                "case_index": br["case_index"],
                "case_id": case.get("case_id"),
                "gold_risk_level": g,
                "base_json_valid": br["json_valid"],
                "lora_json_valid": lr["json_valid"],
                "base_schema_full_ok": br["schema_contract_ok"],
                "lora_schema_full_ok": lr["schema_contract_ok"],
                "base_schema_core_ok": br["schema_core_ok"],
                "lora_schema_core_ok": lr["schema_core_ok"],
                "base_risk_match": rm_b if g else None,
                "lora_risk_match": rm_l if g else None,
                "base_escalation_match": esc_b,
                "lora_escalation_match": esc_l,
                "base_risk_level": br.get("risk_level_normalized"),
                "lora_risk_level": lr.get("risk_level_normalized"),
            }
        )

    report: dict[str, Any] = {
        "base_model": args.base_model,
        "adapter_dir": str(adapter_path),
        "cases_file": str(cases_path),
        "max_new_tokens": args.max_new_tokens,
        "prompt_note": "User JSON excludes eval `expected` gold labels (no leakage).",
        "extraction": "balanced_brace_json_with_fallback",
        "json_parse": {"base": base_parse, "lora": lora_parse},
        "training_schema_contract_full": {"base": base_schema, "lora": lora_schema},
        "schema_core_relaxed": {"base": base_core, "lora": lora_core},
        "risk_level_exact_match_vs_gold": {
            "total_with_gold": n_gold_risk,
            "base_correct": base_risk_correct,
            "lora_correct": lora_risk_correct,
            "base_rate": base_risk_correct / max(n_gold_risk, 1),
            "lora_rate": lora_risk_correct / max(n_gold_risk, 1),
            "delta_correct": lora_risk_correct - base_risk_correct,
        },
        "escalation_match_vs_gold": {
            "total_with_gold": n_gold_esc,
            "base_correct": base_esc_correct,
            "lora_correct": lora_esc_correct,
            "base_rate": base_esc_correct / max(n_gold_esc, 1),
            "lora_rate": lora_esc_correct / max(n_gold_esc, 1),
            "delta_correct": lora_esc_correct - base_esc_correct,
        },
        "delta_json_valid": lora_parse["pass"] - base_parse["pass"],
        "delta_schema_full_pass": lora_schema["pass"] - base_schema["pass"],
        "delta_schema_core_pass": lora_core["pass"] - base_core["pass"],
        "per_case": per_case,
    }

    # Drop heavy parsed blobs from row copies if re-serializing — strip for json output
    def slim_rows(rows: list[dict[str, Any]]) -> None:
        for r in rows:
            r.pop("parsed", None)

    slim_rows(base_rows)
    slim_rows(lora_rows)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rm = report["risk_level_exact_match_vs_gold"]
    esc = report["escalation_match_vs_gold"]

    md_lines = [
        "# MindForge — Base vs LoRA evaluation",
        "",
        "User prompts **exclude** `expected` gold labels from the eval JSON (fair comparison).",
        "",
        f"- **Cases:** `{cases_path}` ({len(cases)} rows)",
        f"- **Base model:** `{args.base_model}`",
        f"- **Adapter:** `{adapter_path}`",
        f"- **max_new_tokens:** {args.max_new_tokens}",
        "",
        "## Metrics",
        "",
        "| Metric | Base | + LoRA | Δ (LoRA − Base) |",
        "|--------|------|--------|-----------------|",
        f"| JSON parse | {base_parse['pass']}/{base_parse['total']} ({base_parse['rate']:.2%}) | "
        f"{lora_parse['pass']}/{lora_parse['total']} ({lora_parse['rate']:.2%}) | "
        f"{report['delta_json_valid']:+d} cases |",
        f"| **Full SFT schema** (13 keys, train JSONL) | {base_schema['pass']}/{base_schema['total']} ({base_schema['rate']:.2%}) | "
        f"{lora_schema['pass']}/{lora_schema['total']} ({lora_schema['rate']:.2%}) | "
        f"{report['delta_schema_full_pass']:+d} cases |",
        f"| **Core schema** (risk + score/band + disclaimer) | {base_core['pass']}/{base_core['total']} ({base_core['rate']:.2%}) | "
        f"{lora_core['pass']}/{lora_core['total']} ({lora_core['rate']:.2%}) | "
        f"{report['delta_schema_core_pass']:+d} cases |",
        f"| **Gold risk_level** exact match | {rm['base_correct']}/{rm['total_with_gold']} ({rm['base_rate']:.2%}) | "
        f"{rm['lora_correct']}/{rm['total_with_gold']} ({rm['lora_rate']:.2%}) | "
        f"{rm['delta_correct']:+d} cases |",
        f"| **Gold escalation** match (heuristic) | {esc['base_correct']}/{esc['total_with_gold']} ({esc['base_rate']:.2%}) | "
        f"{esc['lora_correct']}/{esc['total_with_gold']} ({esc['lora_rate']:.2%}) | "
        f"{esc['delta_correct']:+d} cases |",
        "",
        "## Per case",
        "",
        "| # | case_id | gold risk | base risk OK | LoRA risk OK | base core | LoRA core |",
        "|---|---------|-----------|--------------|--------------|-----------|-----------|",
    ]
    for row in per_case:
        gid = row.get("case_id") or "—"
        gr = row.get("gold_risk_level") or "—"
        brm = row.get("base_risk_match")
        lrm = row.get("lora_risk_match")
        brs = "✓" if row.get("base_schema_core_ok") else "✗"
        lrs = "✓" if row.get("lora_schema_core_ok") else "✗"
        md_lines.append(
            f"| {row['case_index']} | {gid} | {gr} | {brm} | {lrm} | {brs} | {lrs} |"
        )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(report["risk_level_exact_match_vs_gold"], indent=2))
    print(json.dumps(report["training_schema_contract_full"], indent=2))
    print(json.dumps(report["schema_core_relaxed"], indent=2))
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
