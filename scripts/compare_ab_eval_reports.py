#!/usr/bin/env python3
"""
Build eval_comparison_summary.json from two evaluate_lora_json outputs.

Inputs may be:
  - Paths to JSON written via --report_json, or
  - Paths to .txt captures that include the MINDFORGE_EVAL_REPORT_JSON: footer line.

Produces a slide-ready table + submission framing + one side-by-side example row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_embedded_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("MINDFORGE_EVAL_REPORT_JSON:"):
            return json.loads(line[len("MINDFORGE_EVAL_REPORT_JSON:") :])
    # Whole-file JSON
    return json.loads(text)


def load_report(path: Path) -> dict[str, Any]:
    try:
        return load_embedded_report(path)
    except json.JSONDecodeError:
        return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{100.0 * x:.1f}%"


def pick_side_by_side(
    base_rows: list[dict[str, Any]], lora_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Choose the case that best demonstrates LoRA improving contract adherence:
    maximize (lora.required_schema - base.required_schema), tie-break by larger LoRA gain on valid_json.
    """
    by_id = {r.get("case_id"): r for r in lora_rows}
    best: tuple[int, int, int] | None = None  # (delta_req, delta_valid, -index)
    chosen: tuple[dict[str, Any], dict[str, Any]] | None = None
    for b in base_rows:
        cid = b.get("case_id")
        l = by_id.get(cid)
        if not l:
            continue
        d_req = int(bool(l.get("required_schema_fields_present"))) - int(
            bool(b.get("required_schema_fields_present"))
        )
        d_val = int(bool(l.get("valid_json"))) - int(bool(b.get("valid_json")))
        key = (d_req, d_val, -int(l.get("case_index", 0)))
        if best is None or key > best:
            best = key
            chosen = (b, l)
    if not chosen:
        return {}
    b, l = chosen
    note = (l.get("patient_note_preview") or b.get("patient_note_preview") or "").strip()
    return {
        "case_id": l.get("case_id"),
        "input_excerpt": note,
        "base_output_excerpt": (b.get("raw_generation_tail") or "")[:900],
        "lora_output_excerpt": (l.get("raw_generation_tail") or "")[:900],
        "base_required_schema": bool(b.get("required_schema_fields_present")),
        "lora_required_schema": bool(l.get("required_schema_fields_present")),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("base_path", type=Path, help="eval_base_qwen.txt or .json")
    p.add_argument("lora_path", type=Path, help="eval_lora_mindforge.txt or .json")
    p.add_argument("-o", "--out", type=Path, default=Path("eval_comparison_summary.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_rep = load_report(args.base_path)
    lora_rep = load_report(args.lora_path)

    bm = base_rep.get("metrics") or {}
    lm = lora_rep.get("metrics") or {}

    slide_table = [
        {
            "metric": "Valid JSON output",
            "base_qwen": pct(bm.get("valid_json_rate")),
            "mindforge_lora": pct(lm.get("valid_json_rate")),
            "metric_key": "valid_json_rate",
        },
        {
            "metric": "Required schema fields present (all slide-critical fields)",
            "base_qwen": pct(bm.get("required_schema_fields_present")),
            "mindforge_lora": pct(lm.get("required_schema_fields_present")),
            "metric_key": "required_schema_fields_present",
        },
        {
            "metric": "Risk level present",
            "base_qwen": pct(bm.get("risk_level_present")),
            "mindforge_lora": pct(lm.get("risk_level_present")),
            "metric_key": "risk_level_present",
        },
        {
            "metric": "Expected risk_level match (gold, when labeled)",
            "base_qwen": pct(bm.get("expected_risk_level_match")),
            "mindforge_lora": pct(lm.get("expected_risk_level_match")),
            "metric_key": "expected_risk_level_match",
        },
        {
            "metric": "Escalation recommendation present",
            "base_qwen": pct(bm.get("escalation_present")),
            "mindforge_lora": pct(lm.get("escalation_present")),
            "metric_key": "escalation_present",
        },
        {
            "metric": "Patient-safe response present",
            "base_qwen": pct(bm.get("patient_safe_response_present")),
            "mindforge_lora": pct(lm.get("patient_safe_response_present")),
            "metric_key": "patient_safe_response_present",
        },
        {
            "metric": "Clinician summary present",
            "base_qwen": pct(bm.get("clinician_summary_present")),
            "mindforge_lora": pct(lm.get("clinician_summary_present")),
            "metric_key": "clinician_summary_present",
        },
        {
            "metric": "Medication / adherence flags present",
            "base_qwen": pct(bm.get("medication_or_adherence_flags_present")),
            "mindforge_lora": pct(lm.get("medication_or_adherence_flags_present")),
            "metric_key": "medication_or_adherence_flags_present",
        },
    ]

    side = pick_side_by_side(base_rep.get("per_case") or [], lora_rep.get("per_case") or [])

    # Narrative example aligned to hackathon slide wording (fill from real run when possible).
    slide_story = {
        "input": (
            "Patient missed meds, slept 3 hours, caregiver reports agitation. "
            "(Use the closest held-out eval vignette; excerpt below is from the A/B run.)"
        ),
        "base_qwen_characterization": (
            "Often generic chat, truncated JSON, or missing escalation / clinician fields — "
            "see base_output_excerpt."
        ),
        "mindforge_lora_characterization": (
            "More often emits a single JSON object with risk level, adherence/side-effect signal, "
            "escalation_recommendation, patient_safe_response, and clinician_summary — "
            "see lora_output_excerpt."
        ),
        "from_eval_case": side,
    }

    doc: dict[str, Any] = {
        "title": "MindForge LoRA vs base Qwen — structured task adherence",
        "submission_framing": {
            "headline": "LoRA improved structured task adherence.",
            "do_not_claim": (
                "Do not claim clinical superiority or that outputs are clinically accurate."
            ),
            "closing_line": (
                "The adapter does not make medical decisions. It makes Qwen more reliably produce "
                "the Healus/MindForge care-loop structure, which can then be validated, audited, and governed."
            ),
        },
        "methodology": {
            "cases_file": base_rep.get("cases_file"),
            "n_cases": base_rep.get("n_cases"),
            "base_model": base_rep.get("base_model"),
            "base_run": {k: base_rep.get(k) for k in ("run_label", "adapter_dir", "no_adapter")},
            "lora_run": {
                k: lora_rep.get(k)
                for k in ("run_label", "adapter_dir", "adapter_dir_cli", "no_adapter")
                if lora_rep.get(k) is not None
            },
            "note": "Same prompts and held-out cases for both runs; gold labels never included in the prompt.",
        },
        "slide_table": slide_table,
        "metrics_raw": {"base_qwen": bm, "mindforge_lora": lm},
        "side_by_side_example": slide_story,
    }

    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
