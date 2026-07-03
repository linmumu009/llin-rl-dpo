#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
DATASET_PATH="${DATASET_PATH:-/workspace/llin-rl-dpo/datasets/tiny_dpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-smoke}"
MAX_STEPS="${MAX_STEPS:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29617}"

export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MODELSCOPE_OFFLINE="${MODELSCOPE_OFFLINE:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export NPROC_PER_NODE
export MASTER_PORT

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
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --max_steps "${MAX_STEPS}" \
  --logging_steps 1 \
  --save_strategy no \
  --eval_strategy no \
  --output_dir "${OUTPUT_DIR}" \
  --check_model false \
  --fsdp fsdp2
