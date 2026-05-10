# MindForge AI — Fine-Tuning Step 2

## Goal
Run a fast LoRA SFT on the synthetic MindForge dataset to teach the model:

- risk level output contract
- JSON structure
- adherence flags
- escalation language
- patient-safe and clinician-ready summaries

This is hackathon-grade fine-tuning, not production clinical validation.

## Default Model

Use `Qwen/Qwen2.5-0.5B-Instruct` first for speed. If time remains, repeat with `Qwen/Qwen2.5-1.5B-Instruct`.

## Commands

```bash
ssh root@YOUR_DROPLET_IP

apt update && apt install -y git python3-venv python3-pip tmux

git clone https://github.com/ORG/mindforge-ai-amd-hackathon.git
cd mindforge-ai-amd-hackathon

tmux new -s mindforge

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

Check GPU:

```bash
amd-smi list || true
python scripts/check_amd_gpu.py
```

Install ROCm PyTorch if torch is missing or GPU is not visible. Match the wheel to the droplet's ROCm version.

Example for ROCm 7.0:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0 --upgrade --force-reinstall
```

Install training dependencies:

```bash
pip install -r training/requirements-rocm.txt
```

Login to Hugging Face:

```bash
huggingface-cli login
```

Run LoRA SFT:

```bash
python training/train_qwen_lora_trl.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --train_file data/synthetic/mindforge_train.jsonl \
  --validation_file data/synthetic/mindforge_validation.jsonl \
  --output_dir outputs/mindforge-qwen-lora \
  --epochs 6 \
  --batch_size 1 \
  --grad_accum 8 \
  --learning_rate 2e-4
```

Smoke-test inference:

```bash
python training/infer_lora.py \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_dir outputs/mindforge-qwen-lora
```

Evaluate a few JSON outputs:

```bash
python scripts/evaluate_lora_json.py \
  --cases data/synthetic/mindforge_eval_cases.json \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter_dir outputs/mindforge-qwen-lora \
  --limit 5
```

**Before / after (base vs LoRA, gold labels, no leakage):** prompts omit eval `expected` fields. Reports JSON parse, full SFT schema, relaxed “core” schema, and match vs `expected.risk_level` / escalation.

```bash
python scripts/evaluate_base_vs_lora.py \
  --cases data/synthetic/mindforge_eval_cases.json \
  --adapter_dir outputs/mindforge-qwen-lora
```

Writes `outputs/base_vs_lora_report.json` and `outputs/base_vs_lora_report.md`.

Run local Gradio demo:

```bash
python run_demo.py
# or: python -m app.gradio_app_minimal
```

Open:

```text
http://YOUR_DROPLET_IP:7860
```

**Public URL (hackathon judges):**

1. **Droplet IP + port** — App listens on `0.0.0.0:7860`. In **DigitalOcean → Networking → Firewalls**, allow **inbound TCP 7860** (or your `PORT`) to the droplet, then share `http://<public-ipv4>:7860`.
2. **Gradio share link (no firewall change)** — Run `GRADIO_SHARE=1 python run_demo.py`. The terminal prints a temporary `https://*.gradio.live` URL (valid while the process runs).
3. **Production-style** — Put **nginx + HTTPS** (Let’s Encrypt) in front of port 7860, or deploy a duplicate demo on **Hugging Face Spaces**.

## If training is slow or fails

1. Reduce epochs from 6 to 3.
2. Use `Qwen/Qwen2.5-0.5B-Instruct`, not 1.5B.
3. Set `--max_length 1024`.
4. Keep fine-tuning proof minimal and shift to live demo.

## After training

1. Save screenshot of training logs.
2. Save sample tuned output vs base output.
3. Push adapter or merged model to Hugging Face.
4. Wire the demo app to Hugging Face Space.
5. Record demo.
6. Submit early.
