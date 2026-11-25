set -o pipefail
cd /workspace/llin-rl-dpo/reference/MindSpeed-MM
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
source /workspace/llin-rl-dpo/.venvs/msmm-qwen36/bin/activate
export PYTHONPATH=/workspace/llin-rl-dpo/reference/MindSpeed-MM:/workspace/llin-rl-dpo/reference/MindSpeed-26.0.0_core_r0.12.1:${PYTHONPATH}
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_NPU_SOCKET_PORT_RANGE=22000-22095
export ASCEND_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CONFIG=/workspace/llin-rl-dpo/configs/rjx_cutoff_test_native_rows0_31_oldenv_sharegpt_2step.yaml
LOG=/workspace/llin-rl-dpo/logs/rjx_cutoff_test_native_rows0_31_oldenv_sharegpt_2step_8npu_20260707.log
EXIT=/workspace/llin-rl-dpo/logs/rjx_cutoff_test_native_rows0_31_oldenv_sharegpt_2step_8npu_20260707.exit
mkdir -p /workspace/llin-rl-dpo/logs
rm -f "$EXIT"
/workspace/llin-rl-dpo/.venvs/msmm-qwen36/bin/python -m torch.distributed.run   --nproc_per_node 8   --nnodes 1   --master_addr localhost   --master_port 29641   mindspeed_mm/fsdp/train/trainer.py "$CONFIG"   2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "$rc" > "$EXIT"
exit "$rc"
