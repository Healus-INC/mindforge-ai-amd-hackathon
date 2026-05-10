#!/usr/bin/env bash
# Clean A/B eval: same held-out cases, base Qwen vs Qwen + MindForge LoRA.
# Writes eval_base_qwen.txt, eval_lora_mindforge.txt, eval_comparison_summary.json at repo root.
#
# Usage (from repo root, GPU recommended):
#   chmod +x scripts/run_mindforge_ab_eval.sh
#   ./scripts/run_mindforge_ab_eval.sh
#
# Override adapter path:
#   ADAPTER_DIR=outputs/mindforge-qwen-lora-mvp1-baseline ./scripts/run_mindforge_ab_eval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
CASES="${CASES:-data/synthetic/mindforge_eval_cases.json}"
LIMIT="${LIMIT:-20}"
ADAPTER_DIR="${ADAPTER_DIR:-outputs/mindforge-qwen-lora-mvp1-baseline}"

# Convenience: point MVP1 path at the demo adapter if you have not created a separate folder.
if [[ ! -d "$ADAPTER_DIR" ]] && [[ -d "outputs/mindforge-qwen-lora" ]]; then
  echo "NOTE: $ADAPTER_DIR missing — symlinking to outputs/mindforge-qwen-lora for this run."
  ln -sfn mindforge-qwen-lora "$ADAPTER_DIR"
fi

echo "=== Base Qwen (no adapter) ==="
python scripts/evaluate_lora_json.py \
  --cases "$CASES" \
  --base_model "$BASE_MODEL" \
  --no_adapter \
  --limit "$LIMIT" \
  --report_json outputs/eval_base_report.json \
  | tee eval_base_qwen.txt

echo "=== MindForge LoRA ==="
python scripts/evaluate_lora_json.py \
  --cases "$CASES" \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$ADAPTER_DIR" \
  --limit "$LIMIT" \
  --report_json outputs/eval_lora_report.json \
  | tee eval_lora_mindforge.txt

python scripts/compare_ab_eval_reports.py \
  eval_base_qwen.txt \
  eval_lora_mindforge.txt \
  -o eval_comparison_summary.json

echo "Done. See eval_comparison_summary.json"
