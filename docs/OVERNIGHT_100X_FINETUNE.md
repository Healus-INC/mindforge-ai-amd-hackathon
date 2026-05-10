# Overnight fine-tune (100× data) — parallel to demo

## Presentation summary — what we trained for (fine-tuning track)

Use this block for slides and judge Q&A.

**Objective:** Teach **Qwen2.5-Instruct** (via **LoRA supervised fine-tuning**) to emit a **single JSON assistant object** for the MindForge care loop: `risk_level`, `risk_score`, concerns and flags, `escalation_recommendation`, patient-safe and clinician-facing text, `care_loop_targets`, `missing_info`, and **`not_medical_advice: true`**. Training data are **synthetic** chat JSONL rows validated against the same contract used at eval time.

**Methodology:** **SFT only** — **PEFT LoRA** on attention + MLP modules, **TRL `SFTTrainer`**, causal LM, **bf16** on GPU, cosine LR schedule. **No GRPO, RLHF, or other RL** on top of SFT.

**Data scale:**

| Split | Demo (small) | 100× scale (cleaned) |
|-------|----------------|----------------------|
| Train | 60 rows (`mindforge_train.cleaned.jsonl`) | **7,200** (`mindforge_train.scale.cleaned.jsonl`) |
| Val | 12 rows | **400** |

The 100× vendor JSON used an alternate assistant shape (`summary`, `key_flags`, nested `escalation`, …). We **`scripts/normalize_100x_assistant_to_sft.py`** map each row to the **canonical 13-key** schema, then **`prepare_sft_dataset.py`** dedupes and reports stratification (`data/synthetic/prepare_sft_dataset_summary.json`).

**Artifacts:**

- **Demo adapter:** `outputs/mindforge-qwen-lora` (default **Qwen2.5-0.5B-Instruct**).
- **Overnight / scale adapter:** `outputs/mindforge-qwen-lora-scale` (default **Qwen2.5-1.5B-Instruct**, `training/run_overnight_scale.sh`).

**Results in the UI:** The Gradio app has a **Training results** tab. It reads **`outputs/training_results_summary.json`** (build with `scripts/build_training_results_summary.py`) and shows **dataset mix** plus **Base vs Base+LoRA** bars from offline eval when reports exist. After the scale run, run `evaluate_base_vs_lora.py` with **`--adapter_dir outputs/mindforge-qwen-lora-scale`** and **`--out_json outputs/base_vs_lora_scale_report.json`**, then rebuild the summary so the chart highlights the **100×** adapter.

---

## Are we “ready”?

You already have:

- **Canonical assistant schema:** `docs/ASSISTANT_JSON_SCHEMA.md`, `prepare_sft_dataset.py`, cleaned JSONL workflow  
- **Eval:** `evaluate_base_vs_lora.py`, gold labels, schema tiers  
- **Separate adapters:** train script writes to **`--output_dir`** — use a **different directory** than the demo adapter  

**Before scaling:**

1. Put **large** train/val JSONL on disk (e.g. `mindforge_train.scale.jsonl` — do **not** overwrite demo files).  
2. **`ingest_scale_zip.py`** copies files and runs **normalize → prepare** into **`.scale.cleaned.jsonl`**, or run **`normalize_100x_assistant_to_sft.py`** + **`prepare_sft_dataset.py`** manually (see below).  
3. Confirm **VRAM** headroom for **1.5B** if you switch base model (see below).  
4. Use **`tmux`** or **`screen`** so SSH drops don’t kill training.

---

## Keep demo vs enable scale experiment

| Asset | Suggested path | Role |
|-------|----------------|------|
| Demo adapter (current) | `outputs/mindforge-qwen-lora` | Judges / stable Gradio default |
| Scale adapter (100×) | `outputs/mindforge-qwen-lora-scale` or `...-1p5b-scale` | Overnight run |

**Gradio switch (same droplet):**

```bash
# Demo (default in code)
BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct ADAPTER_DIR=outputs/mindforge-qwen-lora python run_demo.py

# After scale run succeeds
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct ADAPTER_DIR=outputs/mindforge-qwen-lora-scale python run_demo.py
```

`BASE_MODEL` **must** match the base used when training that adapter.

---

## Model size: 0.5B vs 1.5B for 100× data

| | **Qwen2.5-0.5B-Instruct** | **Qwen2.5-1.5B-Instruct** |
|--|--------------------------|---------------------------|
| **Fit** | LoRA can absorb more **diverse** patterns with **more data**, but capacity is smaller | **Better** if labels are noisy/complex or you want headroom |
| **Speed / VRAM** | Faster, lighter | ~3× larger weights; longer steps; watch **OOM** |
| **Recommendation** | Fine if budget tight and JSON schema stays strict | **Reasonable default for a serious 100× run** on a **MI300X-class** GPU |

You can **train 0.5B overnight first** as a cheap checkpoint, then **1.5B** if metrics plateau — **keep separate `output_dir`** for each.

---

## End-to-end (overnight)

### 0. Ingest the 100× zip (recommended)

Copy your zip into the repo (e.g. `/root/hackathon/`), then:

```bash
cd /root/hackathon && source .venv/bin/activate
python scripts/ingest_scale_zip.py /path/to/your_100x_dataset.zip
```

This unpacks, detects `train` / `validation` JSONL by filename, installs:

- `data/synthetic/mindforge_train.scale.jsonl`
- `data/synthetic/mindforge_validation.scale.jsonl`

then runs **`normalize_100x_assistant_to_sft.py`** and **`prepare_sft_dataset.py`** to write **`*.scale.sft.jsonl`** and **`*.scale.cleaned.jsonl`**.

Use **`--no-prepare`** if you only want files copied and will normalize / clean manually.

Zip naming tips: include files like **`mindforge_train.jsonl`** + **`mindforge_validation.jsonl`**, or **`train.jsonl`** + **`validation.jsonl`**.

### 1. Prepare scale data manually (alternative)

```bash
cd /root/hackathon && source .venv/bin/activate

python scripts/prepare_sft_dataset.py \
  --train data/synthetic/mindforge_train.scale.jsonl \
  --validation data/synthetic/mindforge_validation.scale.jsonl \
  --out-train data/synthetic/mindforge_train.scale.cleaned.jsonl \
  --out-validation data/synthetic/mindforge_validation.scale.cleaned.jsonl
```

Create **`mindforge_train.scale.jsonl`** / **`mindforge_validation.scale.jsonl`** by copying or generating — **do not delete** the original demo **`mindforge_train.jsonl`**.

### 2. Start training in `tmux`

```bash
tmux new -s mindforge-scale
cd /root/hackathon && source .venv/bin/activate
export HF_TOKEN=hf_...   # optional, for Hub rate limits

./training/run_overnight_scale.sh
# detach: Ctrl+B then D
```

### 3. Morning checklist

- Inspect **`outputs/.../checkpoint-*`** and **`trainer_state.json`** eval loss.  
- Run **`python scripts/evaluate_base_vs_lora.py`** with **`--base_model`** matching training and **`--adapter_dir outputs/mindforge-qwen-lora-scale`**, **`--out_json outputs/base_vs_lora_scale_report.json`**.  
- Rebuild the presenter JSON: **`python scripts/build_training_results_summary.py`** (writes **`outputs/training_results_summary.json`** for the Gradio **Training results** tab).  
- If good: switch **`ADAPTER_DIR`** / **`BASE_MODEL`** for Gradio; optionally **`hf upload`** the new adapter as a **second** Hub repo or tag.

### 4. A/B eval for slides (base Qwen vs MindForge LoRA, same held-out cases)

Goal: show **structured task adherence** to the care-loop JSON contract — **not** clinical superiority.

From repo root (GPU recommended):

```bash
source .venv/bin/activate
# Base only (no adapter)
python scripts/evaluate_lora_json.py \
  --cases data/synthetic/mindforge_eval_cases.json \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --no_adapter \
  --limit 20 \
  --report_json outputs/eval_base_report.json \
  | tee eval_base_qwen.txt

# MindForge LoRA (use your adapter folder; symlink optional)
python scripts/evaluate_lora_json.py \
  --cases data/synthetic/mindforge_eval_cases.json \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_dir outputs/mindforge-qwen-lora-mvp1-baseline \
  --limit 20 \
  --report_json outputs/eval_lora_report.json \
  | tee eval_lora_mindforge.txt

python scripts/compare_ab_eval_reports.py eval_base_qwen.txt eval_lora_mindforge.txt -o eval_comparison_summary.json
```

One-shot: `./scripts/run_mindforge_ab_eval.sh` (optional env: `ADAPTER_DIR`, `LIMIT`, `BASE_MODEL`).

`eval_comparison_summary.json` includes a **slide_table**, **submission_framing** (wording you asked for), and a **side_by_side_example** keyed off the best contrasting case.

---

### 5. Gradio “Training results” tab

- **Data file:** `outputs/training_results_summary.json` (override with env **`TRAINING_RESULTS_JSON`**).  
- **Refresh:** In the UI, **Training results → Refresh results** after rebuilding the JSON; or restart `run_demo.py`.  
- **What judges see:** Markdown recap (objective, methodology, row counts) + **risk mix** chart for the scale split + **Base vs Base+LoRA** chart when eval JSON is merged (prefers **scale** eval over demo when both exist).

---

## Hugging Face Space + Hub (short)

- **Stable URL:** Space with secrets **`BASE_MODEL`**, **`ADAPTER_DIR`** (or Hub adapter id).  
- **Reproducibility:** Push **demo** adapter to Hub; optional second repo for **scale**.  
Details: **`docs/HF_SPACES_AND_HUB.md`**.
