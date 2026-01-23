#!/bin/bash
set -e

# ===== 不同模型 checkpoint（传给 --model_path）=====
MODELS=(
  # "/root/autodl-tmp/model/step_15"
  "/root/autodl-tmp/model/step_30"
)

# ===== 固定 backbone（不要改）=====
BACKBONE="/root/autodl-tmp/data/models/modelscope_cache/models/Qwen/Qwen3-4B-Base"

# ===== 主测试脚本 =====
SCRIPT="./test_three_datasets.sh"

# ===== 日志目录 =====
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL_PATH")
  LOG_FILE="$LOG_DIR/${MODEL_NAME}.log"

  echo "======================================"
  echo "Running model_path: $MODEL_PATH"
  echo "Backbone: $BACKBONE"
  echo "Log: $LOG_FILE"
  echo "======================================"

  $SCRIPT \
    --model_path "$MODEL_PATH" \
    --backbone "$BACKBONE" \
    2>&1 | tee "$LOG_FILE"

done
