# Hugging Face Hub + Spaces (hackathon deliverable)

## Should you upload the adapter and publish a Space?

**Yes, for most hackathon submissions it helps:**

| Approach | Pros | Cons |
|----------|------|------|
| **Push LoRA adapter to Hub** | Judges reproduce inference; you can load with `PeftModel.from_pretrained(base, hub_id)` on GPU or CPU; clean versioning (tags). | Needs `HF_TOKEN`; adapter + tokenizer files are ~tens of MB. |
| **Hugging Face Space** | Stable **`https://*.hf.space`** URL; built-in HTTPS; easy for judges. | GPU Spaces cost credits / queue; cold start; **base model + adapter** must fit Space runtime (often CPU unless you pick GPU hardware). |

**Practical split:**

1. **Hub:** Push **`outputs/mindforge-qwen-lora`** (demo adapter) + document **`BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct`** in the model card.
2. **Space:** Point `BASE_MODEL`, `ADAPTER_DIR` (or `HF_ADAPTER_REPO`), and `HF_TOKEN` via **Space secrets**. Pin **`requirements.txt`** / ROCm vs CUDA vs CPU per runtime (Spaces often use CUDA or CPU; match your Dockerfile or standard Gradio SDK).

**Demo stability:** Keep the **small demo adapter** on Hub as **`main`** / **`demo`** tag; push **100× scale** adapter as a **separate repo** or tag (e.g. `scale-100x`) so judges always have the known-good demo.

---

## Minimal Hub push (adapter folder)

From repo root, after `hf auth login`:

```bash
hf upload USER_OR_ORG/mindforge-qwen-lora-demo outputs/mindforge-qwen-lora/ --repo-type model
```

Use your real namespace instead of `USER_OR_ORG`. See current Hugging Face CLI docs for `hf upload` flags.

---

## Space secrets (typical)

- `HF_TOKEN` — read private adapters / push if needed  
- `BASE_MODEL` — e.g. `Qwen/Qwen2.5-0.5B-Instruct`  
- `ADAPTER_DIR` — local path if adapter baked into Space repo **or** load from Hub in code  

Spaces often clone your repo; copy **`outputs/mindforge-qwen-lora`** into the Space repo **or** download adapter from Hub at startup in **`run_demo.py`** / **`app/gradio_app_minimal.py`** (small extension if you want Hub-only deployment).

For the **Training results** Gradio tab, commit **`outputs/training_results_summary.json`** (from `scripts/build_training_results_summary.py`) so judges see dataset mix and offline **Base vs LoRA** charts without re-running eval on Space hardware. Override path with secret **`TRAINING_RESULTS_JSON`** if you store the file elsewhere.
