#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
DATASET_PATH="${DATASET_PATH:-/workspace/llin-rl-dpo/datasets/tiny_sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-sft-smoke}"
MAX_STEPS="${MAX_STEPS:-3}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29661}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
FSDP_CONFIG="${FSDP_CONFIG:-fsdp2}"
MAX_LENGTH="${MAX_LENGTH:-512}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TRUNCATION_STRATEGY="${TRUNCATION_STRATEGY:-left}"
USE_LOGITS_TO_KEEP="${USE_LOGITS_TO_KEEP:-false}"
PADDING_FREE="${PADDING_FREE:-false}"
PACKING="${PACKING:-false}"
SEQUENCE_PARALLEL_SIZE="${SEQUENCE_PARALLEL_SIZE:-1}"

export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MODELSCOPE_OFFLINE="${MODELSCOPE_OFFLINE:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export NPROC_PER_NODE
export MASTER_PORT

args=(
  sft
  --model "${MODEL_PATH}"
  --model_type qwen3_5
  --dataset "${DATASET_PATH}"
  --split_dataset_ratio 0
  --dataset_num_proc 1
  --dataloader_num_workers 0
  --torch_dtype bfloat16
  --tuner_type lora
  --target_modules all-linear
  --lora_rank 8
  --lora_alpha 32
  --lora_dropout 0
  --max_length "${MAX_LENGTH}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --logging_steps 1
  --save_strategy "${SAVE_STRATEGY}"
  --eval_strategy "${EVAL_STRATEGY}"
  --output_dir "${OUTPUT_DIR}"
  --check_model false
  --fsdp "${FSDP_CONFIG}"
  --use_logits_to_keep "${USE_LOGITS_TO_KEEP}"
  --padding_free "${PADDING_FREE}"
  --packing "${PACKING}"
  --sequence_parallel_size "${SEQUENCE_PARALLEL_SIZE}"
)

if [[ -n "${TRUNCATION_STRATEGY}" ]]; then
  args+=(--truncation_strategy "${TRUNCATION_STRATEGY}")
fi

swift "${args[@]}"
