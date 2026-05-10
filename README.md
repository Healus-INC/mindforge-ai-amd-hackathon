# MindForge AI — AMD Developer Hackathon

MindForge AI is a focused mental-health risk review copilot built as a Healus module for the AMD Developer Hackathon.

It converts synthetic patient/caregiver notes, medication-adherence signals, device events, sleep/mood changes, and side-effect concerns into structured JSON outputs:

- risk level
- medication/adherence flags
- escalation recommendation
- patient-safe explanation
- clinician-ready summary

This repo is public for hackathon judging and demonstration only. All data is synthetic. No PHI.

## Dataset

See `data/synthetic/README.md`.

## Validation

```bash
python scripts/validate_sft_jsonl.py data/synthetic/mindforge_train.jsonl data/synthetic/mindforge_validation.jsonl
```
