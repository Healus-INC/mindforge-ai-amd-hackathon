#!/usr/bin/env python3
"""
Validate MindForge SFT JSONL, dedupe near-identical rows, and report risk_level stratification.

Does not change row semantics beyond dropping invalid lines and duplicates (keeps first).
Writes cleaned JSONL files for training pipelines.

Usage (from repo root):
  python scripts/prepare_sft_dataset.py \\
    --train data/synthetic/mindforge_train.jsonl \\
    --validation data/synthetic/mindforge_validation.jsonl \\
    --out-train data/synthetic/mindforge_train.cleaned.jsonl \\
    --out-validation data/synthetic/mindforge_validation.cleaned.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from mindforge_schema import training_contract_ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate, dedupe, and stratify-report SFT JSONL.")
    p.add_argument("--train", type=Path, default=Path("data/synthetic/mindforge_train.jsonl"))
    p.add_argument("--validation", type=Path, default=Path("data/synthetic/mindforge_validation.jsonl"))
    p.add_argument(
        "--out-train",
        type=Path,
        default=Path("data/synthetic/mindforge_train.cleaned.jsonl"),
        help="Cleaned training JSONL output path.",
    )
    p.add_argument(
        "--out-validation",
        type=Path,
        default=Path("data/synthetic/mindforge_validation.cleaned.jsonl"),
        help="Cleaned validation JSONL output path.",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Print stats only; do not write output files.",
    )
    return p.parse_args()


def load_assistant_obj(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Return (assistant_dict, error_reason)."""
    try:
        msgs = row["messages"]
        if len(msgs) != 3 or msgs[-1].get("role") != "assistant":
            return None, "bad_messages_shape"
        return json.loads(msgs[-1]["content"]), None
    except Exception as e:
        return None, f"parse_error:{e}"


def dedupe_signature(row: dict[str, Any]) -> str:
    """
    Hash user + assistant payloads so synthetic paraphrases with same clinical JSON collapse.

    Uses canonical JSON for stable ordering.
    """
    msgs = row.get("messages") or []
    user_blob = ""
    asst_blob = ""
    for m in msgs:
        if m.get("role") == "user":
            user_blob = m.get("content", "")
        elif m.get("role") == "assistant":
            asst_blob = m.get("content", "")
    try:
        u_obj = json.loads(user_blob) if user_blob else {}
        a_obj = json.loads(asst_blob) if asst_blob else {}
        canon = json.dumps({"u": u_obj, "a": a_obj}, sort_keys=True, ensure_ascii=False)
    except json.JSONDecodeError:
        canon = json.dumps({"user": user_blob, "assistant": asst_blob}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def risk_level_of_row(row: dict[str, Any]) -> str | None:
    asst, _ = load_assistant_obj(row)
    if not asst:
        return None
    return asst.get("risk_level") if isinstance(asst.get("risk_level"), str) else None


def process_file(
    path: Path,
    *,
    label: str,
) -> tuple[list[dict[str, Any]], Counter[str], list[str]]:
    """Return (valid_rows, risk_counts, drop_reasons_as_log_lines)."""
    if not path.exists():
        print(f"[WARN] Missing {label}: {path}", file=sys.stderr)
        return [], Counter(), [f"missing_file:{path}"]

    seen_sig: set[str] = set()
    kept: list[dict[str, Any]] = []
    risks: Counter[str] = Counter()
    log: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                log.append(f"{path.name}:{i} drop invalid_json {e}")
                continue

            asst, err = load_assistant_obj(row)
            if err:
                log.append(f"{path.name}:{i} drop {err}")
                continue

            ok, reasons = training_contract_ok(asst)
            if not ok:
                log.append(f"{path.name}:{i} drop schema {reasons}")
                continue

            sig = dedupe_signature(row)
            if sig in seen_sig:
                log.append(f"{path.name}:{i} drop duplicate_sig {sig[:12]}…")
                continue
            seen_sig.add(sig)

            kept.append(row)
            rl = asst.get("risk_level", "UNKNOWN")
            risks[rl] += 1

    return kept, risks, log


def print_stratification(title: str, counts: Counter[str]) -> None:
    total = sum(counts.values()) or 1
    print(f"\n=== {title} (n={sum(counts.values())}) ===")
    order = ("LOW", "MODERATE", "HIGH", "CRISIS", "UNKNOWN")
    for k in order:
        if k in counts:
            c = counts[k]
            print(f"  {k:10}  {c:4}  ({100 * c / total:5.1f}%)")
    for k, c in sorted(counts.items()):
        if k not in order:
            print(f"  {k:10}  {c:4}  ({100 * c / total:5.1f}%)")


def main() -> None:
    args = parse_args()
    train_rows, train_risk, train_log = process_file(args.train, label="train")
    val_rows, val_risk, val_log = process_file(args.validation, label="validation")

    print_stratification("TRAIN (cleaned)", train_risk)
    print_stratification("VALIDATION (cleaned)", val_risk)

    if train_log:
        print("\n--- train drops ---")
        for line in train_log[:50]:
            print(line)
        if len(train_log) > 50:
            print(f"... and {len(train_log) - 50} more train lines")
    if val_log:
        print("\n--- validation drops ---")
        for line in val_log[:50]:
            print(line)
        if len(val_log) > 50:
            print(f"... and {len(val_log) - 50} more validation lines")

    summary = {
        "train_in": str(args.train),
        "validation_in": str(args.validation),
        "out_train": str(args.out_train),
        "out_validation": str(args.out_validation),
        "train_rows_kept": len(train_rows),
        "validation_rows_kept": len(val_rows),
        "train_risk_distribution": dict(train_risk),
        "validation_risk_distribution": dict(val_risk),
    }
    out_summary = Path("data/synthetic/prepare_sft_dataset_summary.json")
    if not args.report_only:
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        args.out_train.parent.mkdir(parents=True, exist_ok=True)
        with args.out_train.open("w", encoding="utf-8") as f:
            for row in train_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with args.out_validation.open("w", encoding="utf-8") as f:
            for row in val_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote {args.out_train} ({len(train_rows)} rows)")
        print(f"Wrote {args.out_validation} ({len(val_rows)} rows)")
        print(f"Wrote {out_summary}")
    else:
        print("\n(report-only: no files written)")


if __name__ == "__main__":
    main()
