#!/usr/bin/env bash
set -euo pipefail
cd /workspace/MindSpeed-MM-latest
source /usr/local/Ascend/cann/set_env.sh
source /workspace/msmm-latest-probe-venv/bin/activate
export PYTHONPATH=/workspace/MindSpeed-MM-latest:/workspace/MindSpeed-latest:/workspace/llin-rl-dpo/reference/MindSpeed-LLM:${PYTHONPATH:-}
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPUS_PER_NODE=8
MASTER_ADDR=127.0.0.1
MASTER_PORT=${MASTER_PORT:-6037}
CONFIG=llin_configs/rjx_rows0_31_cutoff4096_2step.yaml
mkdir -p llin_logs
LOG=llin_logs/latest_rjx_rows0_31_cutoff4096_2step_8npu_$(date +%Y%m%d_%H%M%S).log
python -m torch.distributed.run \
  --nproc_per_node ${NPUS_PER_NODE} \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr ${MASTER_ADDR} \
  --master_port ${MASTER_PORT} \
  mindspeed_mm/fsdp/train/trainer.py ${CONFIG} 2>&1 | tee ${LOG}
