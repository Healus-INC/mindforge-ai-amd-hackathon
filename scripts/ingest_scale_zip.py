#!/usr/bin/env python3
"""
Unpack a 100× dataset zip into MindForge scale paths, then optional prepare_sft_dataset.

Expected zip layouts (any subfolder):
  - mindforge_train.scale.jsonl / mindforge_validation.scale.jsonl  (already named)
  - mindforge_train.jsonl / mindforge_validation.jsonl
  - train.jsonl / validation.jsonl (or val.jsonl)

Usage:
  python scripts/ingest_scale_zip.py /path/to/mindforge_100x.zip
  python scripts/ingest_scale_zip.py ./mindforge_100x.zip --no-prepare

Writes:
  data/synthetic/mindforge_train.scale.jsonl
  data/synthetic/mindforge_validation.scale.jsonl

Then runs normalize_100x_assistant_to_sft.py (maps alternate 100× assistant schema to the
canonical SFT contract), then prepare_sft_dataset (unless --no-prepare) to produce:
  data/synthetic/mindforge_train.scale.cleaned.jsonl
  data/synthetic/mindforge_validation.scale.cleaned.jsonl
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

RAW_TRAIN = Path("data/synthetic/mindforge_train.scale.jsonl")
RAW_VAL = Path("data/synthetic/mindforge_validation.scale.jsonl")
SFT_TRAIN = Path("data/synthetic/mindforge_train.scale.sft.jsonl")
SFT_VAL = Path("data/synthetic/mindforge_validation.scale.sft.jsonl")
STAGING = Path("data/synthetic/_scale_zip_staging")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest 100× dataset zip into scale JSONL paths.")
    p.add_argument("zip_path", type=Path, help="Path to .zip file")
    p.add_argument(
        "--no-prepare",
        action="store_true",
        help="Only unpack + rename; do not run prepare_sft_dataset.py",
    )
    return p.parse_args()


def pick_scale_files(jsonl_files: list[Path]) -> tuple[Path | None, Path | None]:
    """Return (train_path, val_path) from extracted paths using filename heuristics."""
    by_lower = {p.name.lower(): p for p in jsonl_files}

    def find_train() -> Path | None:
        for key in (
            "mindforge_train.scale.jsonl",
            "mindforge_train.jsonl",
            "train.jsonl",
            "sft_train.jsonl",
        ):
            if key in by_lower:
                return by_lower[key]
        for p in jsonl_files:
            n = p.name.lower()
            if "train" in n and "valid" not in n:
                return p
        return None

    def find_val() -> Path | None:
        for key in (
            "mindforge_validation.scale.jsonl",
            "mindforge_validation.jsonl",
            "validation.jsonl",
            "val.jsonl",
            "dev.jsonl",
        ):
            if key in by_lower:
                return by_lower[key]
        for p in jsonl_files:
            n = p.name.lower()
            if "valid" in n or n.startswith("val") or "dev" in n:
                return p
        return None

    return find_train(), find_val()


def main() -> None:
    args = parse_args()
    zp = Path(args.zip_path).expanduser()
    if not zp.is_absolute():
        zp = (Path.cwd() / zp).resolve()
    else:
        zp = zp.resolve()
    if not zp.is_file():
        print(f"ERROR: Zip not found: {zp}", file=sys.stderr)
        sys.exit(2)

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zp} -> {STAGING}")
    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(STAGING)

    jsonl_files = sorted(STAGING.rglob("*.jsonl"))
    if not jsonl_files:
        print("ERROR: No .jsonl files found inside zip.", file=sys.stderr)
        sys.exit(3)

    train_src, val_src = pick_scale_files(jsonl_files)
    if train_src is not None and val_src is not None and train_src.resolve() == val_src.resolve():
        print("ERROR: Train and validation resolved to the same file; fix zip contents.", file=sys.stderr)
        sys.exit(5)
    if train_src is None or val_src is None:
        print("Found JSONL files:", *[f"  {p}" for p in jsonl_files], sep="\n")
        print(
            "ERROR: Could not infer train vs validation files. "
            "Rename inside the zip to include train / validation (or val), "
            "or use standard names: mindforge_train.jsonl + mindforge_validation.jsonl",
            file=sys.stderr,
        )
        sys.exit(4)

    RAW_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_src, RAW_TRAIN)
    shutil.copy2(val_src, RAW_VAL)
    print(f"Installed train -> {RAW_TRAIN} ({RAW_TRAIN.stat().st_size} bytes)")
    print(f"Installed val  -> {RAW_VAL} ({RAW_VAL.stat().st_size} bytes)")

    shutil.rmtree(STAGING)

    if args.no_prepare:
        print("Skipping normalize + prepare_sft_dataset (--no-prepare).")
        return

    norm_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "normalize_100x_assistant_to_sft.py"),
        "--in-train",
        str(RAW_TRAIN),
        "--in-val",
        str(RAW_VAL),
        "--out-train",
        str(repo_root / SFT_TRAIN),
        "--out-val",
        str(repo_root / SFT_VAL),
    ]
    print("Running:", " ".join(norm_cmd))
    subprocess.run(norm_cmd, check=True)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "prepare_sft_dataset.py"),
        "--train",
        str(repo_root / SFT_TRAIN),
        "--validation",
        str(repo_root / SFT_VAL),
        "--out-train",
        str(repo_root / "data/synthetic/mindforge_train.scale.cleaned.jsonl"),
        "--out-validation",
        str(repo_root / "data/synthetic/mindforge_validation.scale.cleaned.jsonl"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("\nReady for overnight training:")
    print("  ./training/run_overnight_scale.sh")


if __name__ == "__main__":
    main()
