#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
ADAPTER_PATH="${ADAPTER_PATH:-/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step/v0-20260703-130918/checkpoint-2}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-/workspace/llin-rl-dpo/datasets/tiny_infer.jsonl}"
INFER_BACKEND="${INFER_BACKEND:-pt}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
LOAD_ARGS="${LOAD_ARGS:-false}"

export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MODELSCOPE_OFFLINE="${MODELSCOPE_OFFLINE:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

args=(
  infer
  --model "${MODEL_PATH}"
  --model_type qwen3_5
  --adapters "${ADAPTER_PATH}"
  --load_args "${LOAD_ARGS}"
  --infer_backend "${INFER_BACKEND}"
  --device_map "${DEVICE_MAP}"
  --torch_dtype bfloat16
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --stream false
  --val_dataset "${VAL_DATASET_PATH}"
  --val_dataset_sample 1
)

swift "${args[@]}"
