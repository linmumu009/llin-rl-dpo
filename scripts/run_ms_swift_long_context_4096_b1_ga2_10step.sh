#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
DATASET_PATH="${DATASET_PATH:-/workspace/llin-rl-dpo/datasets/long_context_dpo_192.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-longctx-4096-b1-ga2-10step/${RUN_ID}}"
MASTER_PORT="${MASTER_PORT:-29657}"

DATASET_PATH="${DATASET_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MAX_STEPS=10 \
NUM_TRAIN_EPOCHS=1 \
MAX_LENGTH=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=2 \
TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-left}" \
SAVE_STRATEGY=no \
EVAL_STRATEGY=no \
FSDP_CONFIG=fsdp2 \
MASTER_PORT="${MASTER_PORT}" \
scripts/run_ms_swift_qwen36_dpo_smoke.sh
