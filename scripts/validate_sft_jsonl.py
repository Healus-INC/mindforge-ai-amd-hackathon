#!/usr/bin/env python3
"""Validate JSONL files for supervised fine-tuning (one JSON object per line).

Supports common schemas:
  - Chat: {"messages": [{"role": str, "content": str}, ...]}
  - Alpaca-style: {"instruction": str, "input"?: str, "output": str}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def err(msg: str) -> None:
    print(msg, file=sys.stderr)


def validate_record(obj: Any, path: Path, line_no: int) -> list[str]:
    problems: list[str] = []
    if not isinstance(obj, dict):
        problems.append(f"{path}:{line_no}: record must be a JSON object, got {type(obj).__name__}")
        return problems

    if "messages" in obj:
        msgs = obj["messages"]
        if not isinstance(msgs, list) or len(msgs) == 0:
            problems.append(f"{path}:{line_no}: 'messages' must be a non-empty list")
            return problems
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                problems.append(f"{path}:{line_no}: messages[{i}] must be an object")
                continue
            role = m.get("role")
            content = m.get("content")
            if not isinstance(role, str) or not role.strip():
                problems.append(f"{path}:{line_no}: messages[{i}].role must be a non-empty string")
            if not isinstance(content, str):
                problems.append(f"{path}:{line_no}: messages[{i}].content must be a string")
        return problems

    if "instruction" in obj and "output" in obj:
        for key in ("instruction", "input", "output"):
            if key not in obj:
                continue
            val = obj[key]
            if val is not None and not isinstance(val, str):
                problems.append(f"{path}:{line_no}: '{key}' must be a string or null")
        return problems

    problems.append(
        f"{path}:{line_no}: expected chat ('messages') or alpaca-style ('instruction'+'output') fields"
    )
    return problems


def validate_file(path: Path) -> tuple[int, list[str]]:
    line_no = 0
    count = 0
    all_problems: list[str] = []
    text = path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line_no += 1
        line = raw_line.strip()
        if not line:
            continue
        count += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            all_problems.append(f"{path}:{line_no}: invalid JSON — {e}")
            continue
        all_problems.extend(validate_record(obj, path, line_no))
    return count, all_problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="JSONL files to validate")
    args = parser.parse_args()

    total_records = 0
    problems: list[str] = []
    for p in args.files:
        if not p.is_file():
            err(f"Missing file: {p}")
            return 2
        n, file_problems = validate_file(p)
        total_records += n
        problems.extend(file_problems)

    print(f"Validated {len(args.files)} file(s), {total_records} non-empty JSON line(s).")
    if problems:
        err("\n".join(problems))
        return 1
    print("OK — no schema issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
