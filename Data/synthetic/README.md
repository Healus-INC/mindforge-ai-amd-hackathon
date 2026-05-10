# MindForge AI Synthetic Dataset — Step 1

Created for the AMD Developer Hackathon demo.

## Purpose

This dataset supports a focused Healus/MindForge module:

**Mental Health Risk Review Copilot**

The model receives synthetic patient/caregiver notes, medication-adherence events, sleep/mood signals, side-effect concerns, and device events. It returns a structured JSON risk review for the patient-care loop.

## Public-safety and privacy position

- Synthetic data only.
- No real patient data.
- No PHI.
- No diagnosis or prescribing.
- No replacement for a clinician.
- Crisis cases recommend human/emergency escalation.

## Files

| File | Purpose |
|---|---|
| `mindforge_train.jsonl` | 60 SFT examples in chat/messages format |
| `mindforge_validation.jsonl` | 12 validation examples |
| `mindforge_eval_cases.json` | held-out evaluation cases |
| `mindforge_demo_cases.json` | UI dropdown/demo examples |
| `mindforge_raw_cases.jsonl` | raw input + expected output records |
| `mindforge_output_schema.json` | JSON schema for model output |
| `mindforge_risk_rubric.json` | risk-level rubric |

## Risk levels

- `LOW`: routine self-monitoring
- `MODERATE`: caregiver check-in or routine clinician follow-up if repeated
- `HIGH`: same-day / near-term clinician involvement
- `CRISIS`: immediate emergency/crisis escalation

## Output contract

The assistant must return valid JSON with these top-level fields:

```json
{
  "risk_level": "LOW|MODERATE|HIGH|CRISIS",
  "risk_score": 0,
  "primary_concerns": [],
  "medication_adherence": {
    "status": "",
    "flags": [],
    "recommended_action": ""
  },
  "side_effect_flags": [],
  "sleep_mood_flags": [],
  "safety_flags": [],
  "escalation_recommendation": {
    "level": "self_monitor|caregiver_checkin|clinician_followup|same_day_clinician|emergency_crisis",
    "timeframe": "",
    "reason": ""
  },
  "patient_safe_response": "",
  "clinician_summary": "",
  "care_loop_targets": [],
  "missing_info": [],
  "not_medical_advice": true
}
```

## Hackathon framing

This dataset is intentionally narrow and demo-focused. It is designed to prove the core workflow, not to claim clinical performance.

Recommended pitch sentence:

> MindForge AI turns messy between-visit patient signals into structured risk review, adherence flags, escalation recommendations, patient-safe explanation, and clinician-ready summaries.

Generated: 2026-05-10T00:12:45.131192Z
