# MindForge AI — AMD Hackathon

Dataset and tooling for MindForge risk-review SFT data.

## Dataset layout

Synthetic JSONL files live under `data/synthetic/`:

- `mindforge_train.jsonl`
- `mindforge_validation.jsonl`

## Validate JSONL

After extracting `mindforge_dataset_step1.zip` into this repository root:

```bash
python scripts/validate_sft_jsonl.py \
  data/synthetic/mindforge_train.jsonl \
  data/synthetic/mindforge_validation.jsonl
```

## Git workflow (example)

```bash
git checkout -b dataset-step1
git add .
git commit -m "Add synthetic MindForge risk review dataset"
git push -u origin dataset-step1
```
