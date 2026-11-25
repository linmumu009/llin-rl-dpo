set -e
python -m venv --system-site-packages /workspace/llin-rl-dpo/.venvs/rjx-msmm-native
old=/workspace/llin-rl-dpo/.venvs/msmm-qwen36/lib/python3.11/site-packages
new=/workspace/llin-rl-dpo/.venvs/rjx-msmm-native/lib/python3.11/site-packages
cp -a "$old"/torchdata "$new"/
cp -a "$old"/torchdata-*.dist-info "$new"/
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /workspace/llin-rl-dpo/.venvs/rjx-msmm-native/bin/activate
python - <<'PY'
import importlib, sys
print(sys.executable)
for m in ["torch", "torch_npu", "transformers", "accelerate", "datasets", "torchdata", "triton"]:
    try:
        mod=importlib.import_module(m)
        print(m, getattr(mod, "__version__", "no_version"), getattr(mod, "__file__", ""))
    except Exception as e:
        print(m, "ERR", type(e).__name__, e)
PY