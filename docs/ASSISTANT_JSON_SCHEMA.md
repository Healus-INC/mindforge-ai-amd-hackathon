# MindForge assistant JSON schema (SFT contract)

Every **`messages[-1]`** from the assistant must be **stringified JSON** with exactly these **top-level keys** (same checks as `scripts/validate_sft_jsonl.py` and `scripts/mindforge_schema.py`):

| Key | Type | Notes |
|-----|------|--------|
| `risk_level` | string | One of `LOW`, `MODERATE`, `HIGH`, `CRISIS` |
| `risk_score` | integer | 0–100 |
| `primary_concerns` | array of strings | Non-empty list typical |
| `medication_adherence` | object | e.g. `status`, `flags`, `recommended_action` |
| `side_effect_flags` | array | May be empty |
| `sleep_mood_flags` | array | May be empty |
| `safety_flags` | array | May be empty |
| `escalation_recommendation` | object | e.g. `level`, `timeframe`, `reason` |
| `patient_safe_response` | string | |
| `clinician_summary` | string | |
| `care_loop_targets` | array of strings | e.g. `patient`, `caregiver`, `clinician` |
| `missing_info` | array of strings | May be empty |
| `not_medical_advice` | boolean | Must be **`true`** |

**Do not** introduce alternate top-level keys in training data (e.g. `risk_score_band` only) if you want a single consistent adapter; models learn whatever distribution you emit.

**Escalation object** (`escalation_recommendation`): keep a stable shape in synthetic data, e.g. `level`, `timeframe`, `reason` strings, so evaluation heuristics stay stable.

See also: `data/synthetic/assistant_schema.json` (JSON Schema snapshot).
