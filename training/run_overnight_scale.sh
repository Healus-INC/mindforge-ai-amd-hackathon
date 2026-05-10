#!/usr/bin/env bash
# Overnight LoRA run for large (e.g. 100×) datasets — does NOT touch demo adapter paths unless you set env vars that way.
#
# Usage:
#   chmod +x training/run_overnight_scale.sh
#   tmux new -s ft100
#   ./training/run_overnight_scale.sh
#
# Override any variable inline:
#   BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct TRAIN_FILE=... OUTPUT_DIR=... ./training/run_overnight_scale.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

# --- Defaults: scale preset (change TRAIN_FILE / VAL_FILE to your large JSONL paths) ---
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
export TRAIN_FILE="${TRAIN_FILE:-data/synthetic/mindforge_train.scale.cleaned.jsonl}"
export VAL_FILE="${VAL_FILE:-data/synthetic/mindforge_validation.scale.cleaned.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/mindforge-qwen-lora-scale}"

# Conservative defaults for long runs; tune after first OOM / speed check.
export EPOCHS="${EPOCHS:-3}"
export BATCH="${BATCH:-1}"
export GA="${GA:-8}"
export LR="${LR:-2e-4}"
export MAX_LENGTH="${MAX_LENGTH:-2048}"

echo "=== MindForge overnight training ==="
echo "ROOT=$ROOT"
echo "BASE_MODEL=$BASE_MODEL"
echo "TRAIN_FILE=$TRAIN_FILE"
echo "VAL_FILE=$VAL_FILE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "epochs=$EPOCHS batch=$BATCH grad_accum=$GA lr=$LR max_length=$MAX_LENGTH"
echo ""

if [[ ! -f "$TRAIN_FILE" ]]; then
  echo "ERROR: Train file not found: $TRAIN_FILE"
  echo "Create your 100× JSONL, run prepare_sft_dataset.py with --out-train pointing to .scale.cleaned.jsonl, or set TRAIN_FILE."
  exit 1
fi
if [[ ! -f "$VAL_FILE" ]]; then
  echo "ERROR: Validation file not found: $VAL_FILE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python training/train_qwen_lora_trl.py \
  --base_model "$BASE_MODEL" \
  --train_file "$TRAIN_FILE" \
  --validation_file "$VAL_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH" \
  --grad_accum "$GA" \
  --learning_rate "$LR" \
  --max_length "$MAX_LENGTH"

echo ""
echo "Done. Adapter at: $OUTPUT_DIR"
echo "Demo adapter unchanged at: outputs/mindforge-qwen-lora (if you did not override OUTPUT_DIR to that path)"
