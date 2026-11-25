#!/bin/bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=7200
NPUS_PER_NODE=8
CONFIG=/data/renjunxiang/coding/huawei_train/scripts/train/cutoff_test.yaml
LOGDIR=/data/renjunxiang/coding/huawei_train/logs
mkdir -p $LOGDIR
logfile=smoke_full_nopatch_$(date +%Y%m%d_%H%M%S).log
cd /workspace/MindSpeed-MM
torchrun --nproc_per_node $NPUS_PER_NODE --nnodes 1 --master_addr localhost --master_port 6002 \
  mindspeed_mm/fsdp/train/trainer.py $CONFIG 2>&1 | tee $LOGDIR/$logfile
