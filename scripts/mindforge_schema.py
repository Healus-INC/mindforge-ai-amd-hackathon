"""
Shared MindForge assistant JSON contract (matches validate_sft_jsonl.py).

Use for offline eval: schema completeness vs training labels, not just json.loads success.
"""

from __future__ import annotations

import re
from typing import Any

# Training JSONL assistant object — required keys (see validate_sft_jsonl.py).
REQUIRED_ASSISTANT_KEYS = frozenset(
    {
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
)

VALID_RISK_LEVELS = frozenset({"LOW", "MODERATE", "HIGH", "CRISIS"})

RISK_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRISIS": 3}

# Map eval gold escalation slugs ↔ common model tokens (training used mixed level strings).
_ESCALATION_GROUPS: tuple[frozenset[str], ...] = (
    frozenset(
        {
            "clinician_followup",
            "follow_up",
            "followup",
            "clinical_followup",
            "clinician_follow_up",
        }
    ),
    frozenset({"same_day_clinician", "same_day", "urgent_clinician", "urgent", "sameday"}),
    frozenset({"caregiver_checkin", "caregiver_check_in", "caregiver", "checkin", "check_in"}),
)


def _escalation_bucket(slug: str) -> frozenset[str] | None:
    for g in _ESCALATION_GROUPS:
        if slug in g:
            return g
    return None


def risk_score_numeric_ok(value: Any) -> bool:
    """True if value is an integer 0..100 or a whole float in that range (common LLM quirk)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int) and not isinstance(value, bool):
        return 0 <= value <= 100
    if isinstance(value, float) and value.is_integer():
        i = int(value)
        return 0 <= i <= 100
    return False


def normalize_risk_level(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    s = value.strip().upper()
    if s in VALID_RISK_LEVELS:
        return s
    # Noisy outputs: find first known token (CRISIS before HIGH substring quirks)
    for lvl in ("CRISIS", "HIGH", "MODERATE", "LOW"):
        if lvl in s:
            return lvl
    return None


def prompt_dict_from_eval_case(case: dict[str, Any]) -> dict[str, Any]:
    """
    Build the user payload for inference **without** leaking gold `expected` labels.

    Eval JSON shape: { case_id, input: { patient_note, device_or_app_events }, expected?: ... }
    """
    inp = case.get("input") or {}
    return {
        "case_id": case.get("case_id"),
        "context_type": "synthetic_demo",
        "patient_note": inp.get("patient_note", ""),
        "device_or_app_events": inp.get("device_or_app_events", {}),
        "task": "Return a structured mental-health risk review JSON for the care loop.",
    }


def schema_core_ok(parsed: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """
    Relaxed contract for hackathon eval when models omit long-form lists from training JSONL.

    Requires:
      - valid risk_level enum
      - risk_score int 0..100 OR risk_score_band [low, high]
      - not_medical_advice is True (aligned with training emphasis)
    """
    if not parsed:
        return False, ["not_parsed"]

    reasons: list[str] = []
    if normalize_risk_level(parsed.get("risk_level")) is None:
        reasons.append("invalid_or_missing_risk_level")

    rs = parsed.get("risk_score")
    band = parsed.get("risk_score_band")
    score_ok = risk_score_numeric_ok(rs)
    band_ok = isinstance(band, list) and len(band) == 2 and all(isinstance(x, (int, float)) for x in band)
    if not score_ok and not band_ok:
        reasons.append("need_risk_score_or_band")

    if parsed.get("not_medical_advice") is not True:
        reasons.append("not_medical_advice_not_true")

    return (len(reasons) == 0), reasons


def training_contract_ok(parsed: dict[str, Any] | None) -> tuple[bool, list[str]]:
    """
    True iff parsed JSON satisfies the same structural checks as SFT assistant rows.

    Returns (ok, reasons) where reasons are human-readable issues (empty if ok).
    """
    if not parsed:
        return False, ["not_parsed"]

    reasons: list[str] = []
    missing = sorted(REQUIRED_ASSISTANT_KEYS - set(parsed.keys()))
    if missing:
        reasons.append(f"missing_keys:{missing}")

    rl_norm = normalize_risk_level(parsed.get("risk_level"))
    if rl_norm is None:
        reasons.append(f"invalid_risk_level:{parsed.get('risk_level')!r}")

    rs = parsed.get("risk_score")
    if not risk_score_numeric_ok(rs):
        reasons.append(f"risk_score_not_int_0_100:{rs!r}")

    if parsed.get("not_medical_advice") is not True:
        reasons.append("not_medical_advice_not_true")

    return (len(reasons) == 0), reasons


def escalation_hint(obj: dict[str, Any]) -> str | None:
    """
    Best-effort single string for comparing to gold escalation_level on eval cases.

    Training uses object `escalation_recommendation`; generations sometimes use
    `escalation_level` or nest strings under recommended_action.
    """
    if not obj:
        return None
    if isinstance(obj.get("escalation_level"), str):
        return _slug_escalation(obj["escalation_level"])
    er = obj.get("escalation_recommendation")
    if isinstance(er, str):
        return _slug_escalation(er)
    if isinstance(er, dict):
        for k in ("level", "type", "urgency", "recommendation"):
            v = er.get(k)
            if isinstance(v, str) and v.strip():
                return _slug_escalation(v)
    ra = obj.get("recommended_action")
    if isinstance(ra, dict):
        er2 = ra.get("escalation_recommendation")
        if isinstance(er2, str):
            return _slug_escalation(er2)
    return None


_slug_re = re.compile(r"[^a-z0-9]+")


def _slug_escalation(s: str) -> str:
    s = s.strip().lower()
    s = _slug_re.sub("_", s).strip("_")
    return s


def gold_escalation_slug(case: dict[str, Any]) -> str | None:
    exp = case.get("expected") or {}
    el = exp.get("escalation_level")
    if isinstance(el, str) and el.strip():
        return _slug_escalation(el)
    return None


def escalation_match(parsed: dict[str, Any] | None, case: dict[str, Any]) -> bool | None:
    """None if no gold escalation; else True if prediction matches gold slug or is substring."""
    gold = gold_escalation_slug(case)
    if not gold:
        return None
    if not parsed:
        return False
    pred = escalation_hint(parsed)
    if not pred:
        return False
    if gold == pred or gold in pred or pred in gold:
        return True
    gb = _escalation_bucket(gold)
    pb = _escalation_bucket(pred)
    if gb is not None and gb is pb:
        return True
    return False
