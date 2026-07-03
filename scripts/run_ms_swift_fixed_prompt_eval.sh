#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-/workspace/llin-rl-dpo/datasets/fixed_eval_prompts.jsonl}"
RESULT_PATH="${RESULT_PATH:-/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/result.jsonl}"
INFER_BACKEND="${INFER_BACKEND:-pt}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
LOAD_ARGS="${LOAD_ARGS:-false}"
VAL_DATASET_SAMPLE="${VAL_DATASET_SAMPLE:-}"

export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MODELSCOPE_OFFLINE="${MODELSCOPE_OFFLINE:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$(dirname "${RESULT_PATH}")"

args=(
  infer
  --model "${MODEL_PATH}"
  --model_type qwen3_5
  --load_args "${LOAD_ARGS}"
  --infer_backend "${INFER_BACKEND}"
  --device_map "${DEVICE_MAP}"
  --torch_dtype bfloat16
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --stream false
  --val_dataset "${VAL_DATASET_PATH}"
  --result_path "${RESULT_PATH}"
)

if [[ -n "${ADAPTER_PATH}" ]]; then
  args+=(--adapters "${ADAPTER_PATH}")
fi

if [[ -n "${VAL_DATASET_SAMPLE}" ]]; then
  args+=(--val_dataset_sample "${VAL_DATASET_SAMPLE}")
fi

swift "${args[@]}"
