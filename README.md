# MindForge AI — Mental Health Intelligence (AMD ROCm)

**Hackathon submission:** the canonical branch for judging is **`main`**. On GitHub, set **Settings → General → Default branch** to **`main`** so the landing page matches this README.

**LoRA adapters:** trained weights are in **`outputs/mindforge-qwen-lora/`** (demo-sized run) and **`outputs/mindforge-qwen-lora-scale/`** (overnight scale run) as `adapter_model.safetensors` plus `adapter_config.json` (committed). A single-file **`mindforge-qwen-lora-adapter.zip`** (~223MB) may exist locally for convenience but exceeds GitHub’s per-file limit—clone from this repo or rebuild the zip from those folders.

MindForge AI is a **synthetic hackathon demo** that fine-tunes **Qwen2.5-Instruct** with **LoRA SFT** on structured mental-health-style dialogues, then serves a **Gradio** reviewer UI for human-in-the-loop escalation. **It does not provide medical care**; outputs are for demonstration and offline evaluation only.

This README summarizes **how we used AMD Instinct™ GPUs with ROCm** to run end-to-end fine-tuning and inference, and **what we measured** afterward.

---

## How we leveraged AMD

### Hardware and software stack

Training and inference were run on **AMD Instinct™ MI300X-class** accelerators (reported by `rocm-smi` as **MI300X VF**, **gfx942**) using:

| Layer | Details |
|--------|---------|
| **GPU** | AMD Instinct MI300X VF (MI300X-class, partitioned/virtualized SKU where applicable) |
| **Stack** | **ROCm 7.2** (`rocm-smi` / `amd-smi` reporting **ROCm 7.2.0**) |
| **PyTorch** | **PyTorch built for ROCm** — training artifacts record e.g. `2.13.0.dev…+rocm7.2` (see `outputs/mindforge-qwen-lora-scale/README.md` framework versions) |
| **Training libs** | **TRL** `SFTTrainer`, **PEFT** LoRA, **Transformers**, **bf16** on GPU |

We used the **same open tooling** as the CUDA ecosystem (Hugging Face TRL/PEFT), compiled against **ROCm** so large-matrix work runs on **AMD CDNA** hardware without changing the high-level recipe: **LoRA on attention + MLP**, cosine LR, causal LM objective.

### Why this matters for judging

- **Full fine-tuning loops on AMD**: dataset normalization → JSONL SFT → checkpoints → offline JSON eval → Gradio demo inference — all executed with **GPU-accelerated** forward/backward passes under ROCm.
- **Scale run**: the overnight **100× / scale** pipeline trains **Qwen2.5-1.5B-Instruct** on **7,200** synthetic rows (see `outputs/training_results_summary.json`), **3 epochs**, **2,700 optimizer steps**, with trainer-logged scale below.
- **Throughput signal**: final validation evaluation in Trainer logs shows on the order of **~25.5 samples/sec** on the eval split at checkpoint (machine-dependent; see `trainer_state.json` `eval_*` fields).

### Training compute snapshot (scale adapter, final checkpoint)

From `outputs/mindforge-qwen-lora-scale/checkpoint-2700/trainer_state.json` (representative of training on AMD):

| Metric | Value |
|--------|--------|
| **Global steps** | 2,700 (3 epochs, `max_steps` aligned with recipe) |
| **Trainer token counter (final logged step)** | ~**11.5M** tokens processed through training logs |
| **Approx. cumulative FLOPs** (Transformers estimate) | ~**9.2 × 10¹⁶** (`total_flos` — relative scale, not billing) |
| **Final train loss** (last step) | ~**0.057** |
| **Final eval loss** | ~**0.055** |
| **Final eval mean token accuracy** | ~**0.977** |

**Tip for hardware screenshots:** during training, **`rocm-smi`** or **`amd-smi monitor`** captures utilization for slides; the JSON above captures **algorithmic** convergence and Trainer-reported throughput.

---

## Results (offline structured JSON eval)

Summaries are merged into **`outputs/training_results_summary.json`** (regenerate with `python scripts/build_training_results_summary.py` after eval).

### Scale adapter — `Qwen2.5-1.5B` + `outputs/mindforge-qwen-lora-scale`

Held-out suite: **8** cases (`data/synthetic/mindforge_eval_cases.json`).

| Metric | Base | + LoRA |
|--------|------|--------|
| JSON parse success | 87.5% | 87.5% |
| **Core schema adherence** | 25% | **62.5%** |
| Full training schema (strict) | 25% | 0% |
| Risk level exact match | 12.5% | 12.5% |
| Escalation match | 0% | 12.5% |

**Takeaway:** LoRA training on AMD materially improves **structured “core” JSON adherence** vs base on this suite; the strict full schema remains challenging on small N — consistent with a demo-focused SFT pass and conservative parsing rules.

### Demo adapter — `Qwen2.5-0.5B` + `outputs/mindforge-qwen-lora`

Same 8 cases: **core schema** **0% → 87.5%** LoRA; **risk level exact match** **0% → 75%** (see summary file for full table). The live UI commonly loads this smaller adapter for latency unless overridden by env.

---

## Repository layout (what judges care about)

| Path | Purpose |
|------|---------|
| `app/gradio_app_minimal.py` | Gradio demo: SYSTEM prompt, JSON extract/sanitize, safety filters |
| `app/training_results_panel.py` | **Training results** tab: markdown + charts from `training_results_summary.json` |
| `training/train_qwen_lora_trl.py` | LoRA SFT entrypoint (TRL) |
| `training/run_overnight_scale.sh` | Scale / overnight defaults |
| `training/requirements-rocm.txt` | ROCm-friendly Python deps |
| `scripts/` | Normalize 100× data, eval, build summary JSON |
| `data/synthetic/` | Synthetic JSONL sources (no PHI) |
| `outputs/training_results_summary.json` | Single-file digest for UI + judges |
| `deploy/mindforge-gradio.service` | **systemd** unit for stable demo hosting |

---

## Quick start (local / GPU machine)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
# Use the PyTorch ROCm wheel index that matches `rocm-smi` (this stack used ROCm 7.2):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install -r training/requirements-rocm.txt
```

Train (demo-sized run example — adjust paths/epochs as needed):

```bash
python training/train_qwen_lora_trl.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --train_file data/synthetic/mindforge_train.cleaned.jsonl \
  --validation_file data/synthetic/mindforge_validation.cleaned.jsonl \
  --output_dir outputs/mindforge-qwen-lora \
  --epochs 6
```

Launch the UI:

```bash
python run_demo.py
```

Environment knobs include `BASE_MODEL`, `ADAPTER_DIR`, `PORT`, `MAX_NEW_TOKENS`, `TRAINING_RESULTS_JSON`. See `docs/FINE_TUNING_STEP2.md` and `docs/OVERNIGHT_100X_FINETUNE.md` for droplet-style workflows.

---

## Deployment (production-style demo)

For a public droplet we typically:

1. Run Gradio on **`0.0.0.0:7860`** (`run_demo.py` / `app.gradio_app_minimal.launch`).
2. Put **Caddy** (or nginx) on **port 80** → `reverse_proxy localhost:7860`.
3. Install **`deploy/mindforge-gradio.service`** under systemd for **restart always** uptime.

Judges can open **`http://<host>/`** without remembering `:7860` if the firewall allows **TCP 80**.

---

## Safety and scope

- **Synthetic data only**; no real patient data.
- UI applies **English-only** sanitization, **unsafe-pattern blocklist**, and **crisis wording compaction** for demo safety.
- **Human review** is assumed for any real-world use; this repo is **not** a medical device.

---

## Documentation index

- `docs/FINE_TUNING_STEP2.md` — ROCm PyTorch install, training, smoke tests  
- `docs/OVERNIGHT_100X_FINETUNE.md` — scale / 100× overnight run  
- `docs/SCALING_FINETUNING.md` — methodology notes  
- `docs/HF_SPACES_AND_HUB.md` — Hugging Face Hub / Spaces alignment  

---

## License

See `LICENSE` in this repository.
