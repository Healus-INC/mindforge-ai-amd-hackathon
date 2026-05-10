# Dataset Submission Notes

## Dataset Claim

MindForge uses a synthetic mental-health risk review dataset designed around between-visit care signals:

- medication adherence
- missed/late doses
- device dispense/retrieval events
- sleep and mood changes
- side-effect concerns
- caregiver observations
- escalation signals

## Why synthetic

This is a healthcare hackathon demo. We intentionally avoid PHI and real patient records.

## How it maps to product

The dataset powers the Healus/MindForge AI module that converts unstructured home/caregiver/device signals into:

- risk level
- medication/adherence flags
- patient-safe explanation
- clinician-ready summary
- escalation recommendation
- audit-friendly JSON output
