#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
DATASET_PATH="${DATASET_PATH:-/workspace/llin-rl-dpo/datasets/tiny_dpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-smoke}"
MAX_STEPS="${MAX_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29617}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
LLIN_SWIFTMODEL_ASSIGN_PATCH="${LLIN_SWIFTMODEL_ASSIGN_PATCH:-0}"
FSDP_CONFIG="${FSDP_CONFIG:-fsdp2}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-}"

export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MODELSCOPE_OFFLINE="${MODELSCOPE_OFFLINE:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export NPROC_PER_NODE
export MASTER_PORT
export LLIN_SWIFTMODEL_ASSIGN_PATCH

if [[ "${LLIN_SWIFTMODEL_ASSIGN_PATCH}" == "1" ]]; then
  export PYTHONPATH="/workspace/llin-rl-dpo/patches:${PYTHONPATH:-}"
fi

args=(
  rlhf
  --rlhf_type dpo
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
  --max_length 512
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 1
  --learning_rate 1e-4
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --logging_steps 1
  --save_strategy "${SAVE_STRATEGY}"
  --save_steps "${SAVE_STEPS}"
  --eval_strategy "${EVAL_STRATEGY}"
  --output_dir "${OUTPUT_DIR}"
  --check_model false
  --fsdp "${FSDP_CONFIG}"
)

if [[ -n "${SAVE_TOTAL_LIMIT}" ]]; then
  args+=(--save_total_limit "${SAVE_TOTAL_LIMIT}")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ -n "${SAVE_ONLY_MODEL}" ]]; then
  args+=(--save_only_model "${SAVE_ONLY_MODEL}")
fi

swift "${args[@]}"
