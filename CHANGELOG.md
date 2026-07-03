# 更新说明

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

