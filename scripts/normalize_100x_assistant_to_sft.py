#!/usr/bin/env python3
"""
Map 100× package assistant JSON (summary/key_flags/escalation/...) to the canonical
MindForge SFT assistant object (see docs/ASSISTANT_JSON_SCHEMA.md).

Reads JSONL chat rows, rewrites assistant message content only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mindforge_schema import normalize_risk_level, training_contract_ok


def map_escalation(src: dict[str, Any]) -> dict[str, Any]:
    e = src.get("escalation") or src.get("escalation_recommendation")
    if isinstance(e, dict) and "level" in e:
        return {
            "level": str(e.get("level", "follow_up")).lower(),
            "timeframe": "as_clinically_appropriate",
            "reason": (e.get("recommendation") or e.get("reason") or e.get("emergency_note") or "")[:800],
        }
    if isinstance(e, dict):
        return {
            "level": str(e.get("level", "clinician_followup")),
            "timeframe": str(e.get("timeframe", "as_clinically_appropriate")),
            "reason": str(e.get("reason", ""))[:800],
        }
    return {"level": "clinician_followup", "timeframe": "as_clinically_appropriate", "reason": ""}


def map_medication_adherence(src: dict[str, Any]) -> dict[str, Any]:
    m = src.get("medication_adherence")
    if not isinstance(m, dict):
        return {"status": "unknown", "flags": [], "recommended_action": ""}
    flags = m.get("flags")
    if not isinstance(flags, list):
        flags = []
    action = m.get("recommended_action") or m.get("action") or ""
    return {
        "status": str(m.get("status", "unknown")),
        "flags": flags,
        "recommended_action": str(action),
    }


def flags_from_key_flags(src: dict[str, Any]) -> tuple[list[str], list[str], list[str], list[str]]:
    """primary_concerns, side_effect, sleep_mood, safety"""
    kf = src.get("key_flags")
    primary: list[str] = []
    se: list[str] = []
    sm: list[str] = []
    sf: list[str] = []
    if isinstance(kf, list):
        for item in kf:
            if not isinstance(item, dict):
                continue
            t = str(item.get("type", "")).lower()
            ev = str(item.get("evidence", "")).strip()
            tag = f"{t}: {ev}" if ev else t
            primary.append(tag)
            if "side" in t or t == "side_effect":
                se.append(ev or tag)
            elif t in ("sleep", "mood", "stress"):
                sm.append(ev or tag)
            elif t in ("safety", "crisis", "self_harm"):
                sf.append(ev or tag)
    if not primary and src.get("summary"):
        primary = [str(src["summary"])[:240]]
    return primary, se, sm, sf


def normalize_assistant(obj: dict[str, Any]) -> dict[str, Any]:
    """
    Convert 100× / alternate layout -> canonical SFT assistant.

    Mapping strategy:
    - `key_flags[]` is flattened into `primary_concerns` and typed buckets (side_effect / sleep_mood / safety).
    - Nested `escalation` becomes flat `escalation_recommendation` {level, timeframe, reason}.
    - `medication_adherence` keeps status but maps `action` -> `recommended_action` for the trainer.
    - `care_loop_actions` is replaced with stable `care_loop_targets` role triple (training shape).
    """
    rl = normalize_risk_level(obj.get("risk_level"))
    rs = obj.get("risk_score")
    if not isinstance(rs, int):
        try:
            rs = int(rs)
        except Exception:
            rs = 50
    rs = max(0, min(100, rs))

    pc, se_flags, sm_flags, safety_flags = flags_from_key_flags(obj)

    out = {
        "risk_level": rl if rl is not None else "MODERATE",
        "risk_score": rs,
        "primary_concerns": pc or ["review_required"],
        "medication_adherence": map_medication_adherence(obj),
        "side_effect_flags": se_flags,
        "sleep_mood_flags": sm_flags,
        "safety_flags": safety_flags,
        "escalation_recommendation": map_escalation(obj),
        "patient_safe_response": str(obj.get("patient_safe_response", ""))[:4000],
        "clinician_summary": str(obj.get("clinician_summary", obj.get("summary", "")))[:4000],
        "care_loop_targets": ["patient", "caregiver", "clinician"],
        "missing_info": [],
        "not_medical_advice": True,
    }
    ok, reasons = training_contract_ok(out)
    if not ok:
        raise ValueError(f"normalize failed: {reasons}")
    return out


def process_line(line: str, lineno: int) -> str | None:
    row = json.loads(line)
    msgs = row["messages"]
    raw = json.loads(msgs[-1]["content"])
    fixed = normalize_assistant(raw)
    msgs[-1]["content"] = json.dumps(fixed, ensure_ascii=False)
    return json.dumps(row, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in-train", type=Path, required=True)
    p.add_argument("--in-val", type=Path, required=True)
    p.add_argument("--out-train", type=Path, required=True)
    p.add_argument("--out-val", type=Path, required=True)
    return p.parse_args()


def convert_file(inp: Path, outp: Path, label: str) -> tuple[int, int]:
    ok_n = err_n = 0
    outp.parent.mkdir(parents=True, exist_ok=True)
    with inp.open("r", encoding="utf-8") as fi, outp.open("w", encoding="utf-8") as fo:
        for i, line in enumerate(fi, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                fo.write(process_line(line, i) + "\n")
                ok_n += 1
            except Exception as e:
                print(f"[{label}] line {i} ERROR {e}", file=sys.stderr)
                err_n += 1
    return ok_n, err_n


def main() -> None:
    args = parse_args()
    tr_ok, tr_err = convert_file(args.in_train, args.out_train, "train")
    va_ok, va_err = convert_file(args.in_val, args.out_val, "val")
    print(json.dumps({"train_out": str(args.out_train), "train_ok": tr_ok, "train_err": tr_err, "val_out": str(args.out_val), "val_ok": va_ok, "val_err": va_err}, indent=2))
    if tr_err or va_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
