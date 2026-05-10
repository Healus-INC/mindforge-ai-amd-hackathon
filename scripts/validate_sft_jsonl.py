#!/usr/bin/env python3
import json
import sys
from pathlib import Path

REQUIRED = {
    "risk_level",
    "risk_score",
    "primary_concerns",
    "medication_adherence",
    "side_effect_flags",
    "sleep_mood_flags",
    "safety_flags",
    "escalation_recommendation",
    "patient_safe_response",
    "clinician_summary",
    "care_loop_targets",
    "missing_info",
    "not_medical_advice",
}

VALID_RISK = {"LOW", "MODERATE", "HIGH", "CRISIS"}

def validate_file(path: Path) -> int:
    errors = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            try:
                row = json.loads(line)
                messages = row["messages"]
                assert len(messages) == 3
                assert messages[0]["role"] == "system"
                assert messages[1]["role"] == "user"
                assert messages[2]["role"] == "assistant"
                assistant = json.loads(messages[2]["content"])
                missing = REQUIRED - set(assistant.keys())
                if missing:
                    raise ValueError(f"missing assistant fields: {sorted(missing)}")
                if assistant["risk_level"] not in VALID_RISK:
                    raise ValueError(f"invalid risk_level: {assistant['risk_level']}")
                if not isinstance(assistant["risk_score"], int) or not (0 <= assistant["risk_score"] <= 100):
                    raise ValueError("risk_score must be int 0..100")
                if assistant["not_medical_advice"] is not True:
                    raise ValueError("not_medical_advice must be true")
            except Exception as e:
                errors += 1
                print(f"[ERROR] {path}:{i}: {e}")
    print(f"{path}: {i if 'i' in locals() else 0} rows checked, {errors} errors")
    return errors

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_sft_jsonl.py data/synthetic/mindforge_train.jsonl data/synthetic/mindforge_validation.jsonl")
        sys.exit(2)
    total = 0
    for p in sys.argv[1:]:
        total += validate_file(Path(p))
    sys.exit(1 if total else 0)

if __name__ == "__main__":
    main()
