# 更新说明

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
