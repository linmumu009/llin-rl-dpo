#!/usr/bin/env bash
set -euo pipefail

ADAPTER_PATH="${ADAPTER_PATH:-/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step/v0-20260703-130918/checkpoint-2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval}"
LOG_DIR="${LOG_DIR:-/workspace/llin-rl-dpo/logs}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
ENABLE_THINKING="${ENABLE_THINKING:-}"
PRESERVE_THINKING="${PRESERVE_THINKING:-}"
TEMPERATURE="${TEMPERATURE:-}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

base_result="${OUTPUT_DIR}/base-${RUN_ID}.jsonl"
adapter_result="${OUTPUT_DIR}/adapter-${RUN_ID}.jsonl"
base_log="${LOG_DIR}/ms_swift_fixed_prompt_eval_base-${RUN_ID}.log"
adapter_log="${LOG_DIR}/ms_swift_fixed_prompt_eval_adapter-${RUN_ID}.log"
base_exit="${LOG_DIR}/ms_swift_fixed_prompt_eval_base-${RUN_ID}.exit"
adapter_exit="${LOG_DIR}/ms_swift_fixed_prompt_eval_adapter-${RUN_ID}.exit"

set +e
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
ENABLE_THINKING="${ENABLE_THINKING}" \
PRESERVE_THINKING="${PRESERVE_THINKING}" \
TEMPERATURE="${TEMPERATURE}" \
RESULT_PATH="${base_result}" \
scripts/run_ms_swift_fixed_prompt_eval.sh >"${base_log}" 2>&1
base_status=$?
printf "%s\n" "${base_status}" >"${base_exit}"

ADAPTER_PATH="${ADAPTER_PATH}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
ENABLE_THINKING="${ENABLE_THINKING}" \
PRESERVE_THINKING="${PRESERVE_THINKING}" \
TEMPERATURE="${TEMPERATURE}" \
RESULT_PATH="${adapter_result}" \
scripts/run_ms_swift_fixed_prompt_eval.sh >"${adapter_log}" 2>&1
adapter_status=$?
printf "%s\n" "${adapter_status}" >"${adapter_exit}"
set -e

printf "base_status=%s\n" "${base_status}"
printf "adapter_status=%s\n" "${adapter_status}"
printf "base_result=%s\n" "${base_result}"
printf "adapter_result=%s\n" "${adapter_result}"
printf "base_log=%s\n" "${base_log}"
printf "adapter_log=%s\n" "${adapter_log}"

if [[ "${base_status}" -ne 0 || "${adapter_status}" -ne 0 ]]; then
  exit 1
fi
