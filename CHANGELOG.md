# 更新说明

## v0.1.4 - 2026-07-03

新增：

- 新增 `reference/ASCEND_QWEN36_FRAMEWORK_SURVEY.md`，专门记录 Ascend + Qwen3.6 + DPO 的框架调查。

框架调查结论：

- 不再只考虑 MindSpeed-RL 和 FSDP。
- 将阿里 ModelScope `ms-swift` 列为下一轮第一优先级，因为官方资料同时覆盖 Qwen3.6、DPO、人类对齐、Ascend NPU、FSDP/FSDP2/DeepSpeed/Megatron。
- 将 LLaMA-Factory NPU 列为第二优先级，因为官方资料覆盖 NPU、DPO 和 Qwen3.6，但仍需实测 `qwen3_5`/Qwen3.6 在 Ascend 上的模型加载。
- 保留 MindSpeed-RL、verl Ascend、FlagScale/FlagOS、vLLM Ascend 作为候选，但分别标注了 DPO/Qwen3.6 直接证据不足或仅适合推理评测的限制。

更新：

- 更新 `README.md` 当前优先级。
- 更新 `reference/FRAMEWORK_EVAL_MATRIX.md`，加入 `ms-swift`、LLaMA-Factory NPU、FlagScale/FlagOS，并调整评估顺序。

## v0.1.3 - 2026-07-03

新增：

- 新增 `scripts/npu_multicard_smoke.py`：逐卡 NPU tensor smoke test。
- 新增 `scripts/hccl_smoke.py`：多进程 HCCL all-reduce smoke test。

容器更新：

- 将 `llin-rl-dpo` 从单逻辑 NPU 改为 8 逻辑 NPU：
  - `ASCEND_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`
  - `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- 容器内 `torch_npu.npu.device_count()` 返回 `8`。
- `npu-smi info` 只显示物理设备 8-15 对应的 NPU 4-7。

验证结果：

- 逐卡 tensor 计算通过：device 0-7 均返回 `4.0`。
- HCCL all-reduce 通过：
  - `torchrun --nproc_per_node 8 --master_port 29591 scripts/hccl_smoke.py`
  - 输出：`world_size=8 all_reduce_sum=36.0 expected=36.0`

训练前置检查：

- Transformers 版本：`4.57.1`
- `AutoConfig.from_pretrained("/models/Qwen3.6-27B", trust_remote_code=True)` 失败。
- 失败原因：Transformers 当前不识别 `model_type=qwen3_5`。
- MindSpeed-LLM FSDP2 支持 DPO 阶段，并支持 `qwen3` / `qwen3-moe` / `qwen3-next`，但当前没有直接支持 `qwen3_5`。
- 服务器已有 qwen3.6-27B vLLM Ascend 推理配方；该配方证明推理可行，但不能直接作为 DPO 训练路径。

结论：

- 多 NPU 训练底座已可用。
- qwen3.6-27B DPO 还不能直接启动，下一步要先解决 `qwen3_5` 模型类支持或选择一个已支持模型做框架基准。

## v0.1.2 - 2026-07-03

修复：

- 解决 `llin-rl-dpo` 容器内 `torch_npu.npu.device_count()` 为 `0` 的问题。
- 根因是 Ascend Docker Runtime 的设备注入依赖 `ASCEND_VISIBLE_DEVICES`，而不是只依赖 `ASCEND_RT_VISIBLE_DEVICES` 或手动 `--device=/dev/davinci*`。
- `ASCEND_VISIBLE_DEVICES=0` 会触发 runtime hook，但该设备已被已有容器占用，容器内 `npu-smi` 报 `device is used`。
- 改用 `ASCEND_VISIBLE_DEVICES=1` 后，容器内 NPU 可见。

当前验证：

- 容器：`llin-rl-dpo`
- 镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64`
- `torch`：`2.7.1+cpu`
- `torch_npu`：`2.7.1.post4`
- `torch_npu.npu.device_count()`：`1`
- 最小 NPU 计算：`torch.ones(4).npu().sum().cpu().item()` 返回 `4.0`

注意：

- 当前只是单 NPU smoke test 通过。
- qwen3.6-27B DPO 训练通常需要多 NPU；正式训练前需要重新确认可用设备集合，例如 `ASCEND_VISIBLE_DEVICES=1,2,3,...`，并避免占用他人容器已分配设备。

## v0.1.1 - 2026-07-03

新增：

- 初始化本地 Git 仓库，添加远程仓库 `https://github.com/linmumu009/llin-rl-dpo.git`，并推送 `v0.1.0` 首版内容。
- 在服务器创建工作区 `/data/liulin/llin-rl-dpo`。
- 按约束只创建我们自己的容器 `llin-rl-dpo`，未修改其他人的容器。
- 分别验证了 MindSpeed-RL 专用镜像和 MindSpeed-LLM 26.0.0 A3 镜像的容器 smoke test。

当前容器：

- 名称：`llin-rl-dpo`
- 镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64`
- 状态：running
- 挂载：
  - `/data/liulin/llin-rl-dpo:/workspace/llin-rl-dpo`
  - `/data/models/Qwen3.6-27B:/models/Qwen3.6-27B:ro`
  - `/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro`
  - `/usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons:ro`

结果：

- MindSpeed-RL 镜像默认 Python 没有 torch，需要激活 `/root/miniconda3/envs/mindspeed_rl_2.2.0`。
- MindSpeed-RL 镜像补充 `LD_LIBRARY_PATH` 后可以 import torch/torch_npu，但 `torch_npu.npu.device_count()` 返回 `0`。
- MindSpeed-LLM 26.0.0 A3 镜像可以 import torch/torch_npu，版本为 torch `2.7.1+cpu`、torch_npu `2.7.1.post4`，但 `torch_npu.npu.device_count()` 仍返回 `0`。
- 单设备和全设备挂载都未解决 device_count 为 0。
- `--privileged` 隔离实验因安全风险未执行。

影响：

- 当前还不能开始 DPO 训练。
- 下一步需要确认 Ascend Docker runtime 的设备访问策略，或由用户明确确认是否允许做 privileged 容器隔离实验。

## v0.1.0 - 2026-07-03

第一版实验台账。

新增：

- 建立项目 README，说明项目来源、服务器边界、评估目标和仓库维护约定。
- 新增 `reference/EXPERIMENT_LOG.md`，记录服务器只读基线检查、MindSpeed-RL DPO 参考信息和下一步计划。
- 新增 `reference/FRAMEWORK_EVAL_MATRIX.md`，定义 MindSpeed-RL、verl NPU/FSDP、torch_npu FSDP 的评估矩阵。
- 新增 `.gitignore`，排除 SSH 登录说明、第三方源码副本、模型权重、checkpoint 和训练输出。

关键结论：

- 服务器是 Ascend 训练环境，`nvidia-smi` 不存在，`npu-smi info` 可用。
- 当前有 8 个 Ascend910 逻辑 NPU，训练进程为空。
- 服务器已有 MindSpeed-RL 2.2.0 A3 ARM 镜像，可作为第一阶段容器基础镜像。
- `/data/models/Qwen3.6-27B` 是 `qwen3_5` conditional/multimodal 配置，不是 MindSpeed-RL 官方 DPO 样例里的普通 `qwen3_30b_a3b`，需要先做架构支持验证。
