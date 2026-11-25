#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/llin-rl-dpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
DATASET_PATH="${DATASET_PATH:-${WORKSPACE_DIR}/datasets/ops_dpo_512.jsonl}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_DIR}/outputs/ms-swift-qwen36-dpo-lrsearch}"
LOG_DIR="${LOG_DIR:-${WORKSPACE_DIR}/logs}"
RUN_ID="${RUN_ID:-lr_${LEARNING_RATE:-1e-4}_$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/${RUN_ID}}"
MAX_STEPS="${MAX_STEPS:-20}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29673}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
FSDP_CONFIG="${FSDP_CONFIG:-${WORKSPACE_DIR}/configs/fsdp2_full_state.json}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${WORKSPACE_DIR}"

export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MODELSCOPE_OFFLINE="${MODELSCOPE_OFFLINE:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export NPROC_PER_NODE
export MASTER_PORT
export PYTHONPATH="${WORKSPACE_DIR}/reference/ms-swift:${PYTHONPATH:-}"

swift rlhf \
  --rlhf_type dpo \
  --model "${MODEL_PATH}" \
  --model_type qwen3_5 \
  --dataset "${DATASET_PATH}" \
  --split_dataset_ratio 0 \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --torch_dtype bfloat16 \
  --tuner_type lora \
  --target_modules all-linear \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_steps 0 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps "${MAX_STEPS}" \
  --logging_steps 1 \
  --save_strategy "${SAVE_STRATEGY}" \
  --eval_strategy no \
  --output_dir "${OUTPUT_DIR}" \
  --check_model false \
  --fsdp "${FSDP_CONFIG}"
