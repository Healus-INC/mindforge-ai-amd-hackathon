# Scaling MindForge fine-tuning (~100× data)

See also **`docs/OVERNIGHT_100X_FINETUNE.md`** (parallel demo vs scale adapters, overnight script) and **`docs/HF_SPACES_AND_HUB.md`** (Hub + Spaces URL).

Before training on much larger JSONL, tighten these areas so compute and quality scale together.

## Base vs LoRA offline check

Run:

```bash
cd /root/hackathon && source .venv/bin/activate
python scripts/evaluate_base_vs_lora.py \
  --cases data/synthetic/mindforge_eval_cases.json \
  --adapter_dir outputs/mindforge-qwen-lora
```

Reports are written to `outputs/base_vs_lora_report.json` and `outputs/base_vs_lora_report.md`.

**What it measures**

1. **JSON parse** — `json.loads` after balanced-brace extraction.
2. **Full SFT schema** — same 13 assistant keys as `validate_sft_jsonl.py` / training JSONL (strict).
3. **Core schema** — relaxed check: valid `risk_level`, integer `risk_score` or `risk_score_band`, `not_medical_advice: true` (good for short eval prompts where models skip long lists).
4. **Gold `risk_level`** — exact match vs `expected.risk_level` in `mindforge_eval_cases.json` (prompts **exclude** `expected` — no label leak).
5. **Gold escalation** — heuristic match vs `expected.escalation_level` (model output shape varies).

Shared helpers live in `scripts/mindforge_schema.py`.

**Interpretation:** Training loss on the JSONL validation split measures **fit to the training distribution**. Offline eval adds **label alignment** and **schema** checks; use a **larger, frozen** eval file as you scale data.

## 1. Data quality and schema

Canonical assistant keys and types: **`docs/ASSISTANT_JSON_SCHEMA.md`** and **`data/synthetic/assistant_schema.json`**.

Before training, run:

```bash
python scripts/prepare_sft_dataset.py
```

This **validates** the full SFT contract, **dedupes** identical user+assistant payloads, prints **risk_level** counts (stratification report), and writes **`mindforge_*.cleaned.jsonl`**. Point **`train_qwen_lora_trl.py`** at the `.cleaned.jsonl` files when you regenerate data.

- **Lock one assistant JSON schema** (your `validate_sft_jsonl.py` contract) and reject or repair rows that drift before training.
- **Dedupe** near-duplicate `messages` (hash user+assistant content) so epoch × batch does not overfit repeated vignettes.
- **Stratify `risk_level`** (LOW / MODERATE / HIGH / CRISIS) so rare escalation patterns appear often enough to learn.
- **Hold out** a frozen validation set (and optional tiny eval set with labels) never merged into training—reuse it for every run to compare checkpoints.

## 2. Metrics beyond JSON validity

- **Train/eval loss** on tokens is necessary but not sufficient.
- Add cheap offline checks: **JSON schema pass rate**, **required-field presence**, optional **risk-label accuracy** if you store gold `risk_level` on eval cases.
- Compare **base vs LoRA** on the same fixed suite (`scripts/evaluate_base_vs_lora.py`) after each training milestone.

## 3. Training configuration

- **Epochs:** With 100× rows, prefer **fewer epochs** or **early stopping on eval_loss** to avoid overfitting small-pattern quirks.
- **Learning rate / LoRA rank:** Large data often tolerates **same or slightly lower LR**; consider increasing **`lora_r`** only if underfitting.
- **Sequence length:** Align **`max_length`** with your longest realistic assistant JSON + chat overhead; trim noisy fluff in data instead of blindly raising length.
- **Batch / grad accumulation:** Increase effective batch size only if stable on ROCm OOM budget.

## 4. Inference robustness (reduces “pretty metrics, ugly demos”)

- **Truncation** causes invalid JSON at inference—raise **`max_new_tokens`** (repo defaults target **~2048** for full MindForge JSON) or shorten prompts; optionally add **repair / constrained decoding** later. This is an **inference** knob—**no retrain** required unless training rows were systematically truncated by **`max_length`**.
- Keep **system prompt** aligned between training and inference (Gradio / API).

## 5. Ops and reproducibility

- **Pin** ROCm PyTorch index + library versions (requirements file + lock).
- **Snapshot data** with manifest (hash, row counts, split names).
- **Version adapters** (`outputs/…`, Hub tags) and store **`trainer_state.json`** next to each run.

## 6. When to expand scope

- If JSON validity plateaus, fix **format consistency in labels** before adding more rows.
- If escalation behavior is wrong despite valid JSON, add **supervised examples** for crisis/high-risk spans and **balance** those slices.
