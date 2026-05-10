Hackathon proof bundle (generated on droplet)
============================================

Logs and summaries saved here for README / pitch deck copy-paste:

  training_metrics_summary.txt   — eval_loss, eval_mean_token_accuracy, epoch 6 (from trainer_state.json)
  amd_gpu_check.txt             — AMD Instinct MI300X VF, ROCm PyTorch, matmul OK
  ../infer_smoke_proof.txt      — single-case LoRA inference JSON + JSON_VALID=true
  ../eval_cases_proof.txt       — evaluate_lora_json.py run (8 cases in dataset; limit 10)

Screenshots you should still capture locally (IDE / tmux scrollback):
  (1) Training completion table from your original training session (your metrics are archived in trainer_state.json).
  (2) Terminal line "Saved LoRA adapter to: outputs/mindforge-qwen-lora" from training stdout.
  (7) Optional: full-screen amd-smi or python scripts/check_amd_gpu.py — see amd_gpu_check.txt text.

Gradio: demo is running at http://YOUR_DROPLET_IP:7860 (public IP was 129.212.177.83 at setup time).
Open in your browser and screenshot the MindForge AI page after clicking "Run risk review".

Adapter backup zip (repo root):
  /root/hackathon/mindforge-qwen-lora-adapter.zip
