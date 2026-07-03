#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
DATASET_PATH="${DATASET_PATH:-/workspace/llin-rl-dpo/datasets/ops_dpo_512.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-ops-100step/${RUN_ID}}"
MAX_STEPS="${MAX_STEPS:-100}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
MASTER_PORT="${MASTER_PORT:-29643}"
FSDP_CONFIG="${FSDP_CONFIG:-/workspace/llin-rl-dpo/configs/fsdp2_full_state.json}"

DATASET_PATH="${DATASET_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MAX_STEPS="${MAX_STEPS}" \
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
SAVE_STRATEGY=steps \
SAVE_STEPS="${MAX_STEPS}" \
SAVE_TOTAL_LIMIT=1 \
SAVE_ONLY_MODEL=true \
FSDP_CONFIG="${FSDP_CONFIG}" \
MASTER_PORT="${MASTER_PORT}" \
scripts/run_ms_swift_qwen36_dpo_smoke.sh
