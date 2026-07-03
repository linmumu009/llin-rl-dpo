# llin-rl-dpo

这是 qwen3.6-27B 在 Ascend 服务器上做 DPO/RL 训练框架评估的实验仓库。

## 我们是怎么来的

这个项目从一次服务器可行性检查开始：

- 目标服务器已经可以通过 SSH 私钥登录。
- 服务器是 Ascend 训练环境，不是 NVIDIA CUDA 环境。
- 当前目标模型目录是 `/data/models/Qwen3.6-27B`。
- 目标任务是 DPO。
- 用户约束是：不改动物理机，不动别人的 Docker，只新建并使用我们自己的容器 `llin-rl-dpo`。

本仓库用于记录：

- 每一步环境配置顺序。
- 每个框架的跑通情况。
- 训练效率指标。
- 训练效果指标。
- 每个版本和上一个版本的区别。

## 当前服务器基线

已确认的信息见 [reference/EXPERIMENT_LOG.md](reference/EXPERIMENT_LOG.md)。

摘要：

- Docker `28.5.2`
- Ascend Docker runtime 已存在。
- `npu-smi info` 可用。
- 8 个 Ascend910 逻辑 NPU，设备文件从 `/dev/davinci0` 到 `/dev/davinci15`。
- `/data` 剩余约 1.3T。
- 服务器已有 MindSpeed-RL 镜像：
  `swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed_rl_pt25_25rc3:2.2.0-A3-ARM`

## 当前关键判断

MindSpeed-RL 是第一优先级，因为服务器已有专用镜像，而且官方仓库提供 DPO 入口。

但当前 `qwen3.6-27B` 的 `config.json` 显示它不是普通 Qwen3 dense 模型，而是：

- `architectures`: `Qwen3_5ForConditionalGeneration`
- `model_type`: `qwen3_5`
- `language_model_only`: `false`
- 包含 `text_config` 和 `vision_config`

所以不能把 MindSpeed-RL 的 `qwen3_30b_a3b` DPO 样例直接照抄到 qwen3.6-27B。第一阶段必须先做环境和架构支持 smoke test。

## 评估框架

详见 [reference/FRAMEWORK_EVAL_MATRIX.md](reference/FRAMEWORK_EVAL_MATRIX.md)。

当前优先级：

1. MindSpeed-RL DPO
2. verl NPU / FSDP
3. torch_npu 原生 FSDP

## 版本说明

版本记录见 [CHANGELOG.md](CHANGELOG.md)。

当前版本：

- `v0.1.0`：建立实验仓库、完成服务器只读基线检查、拉取 MindSpeed-RL 作为本地参考资料、记录第一版评估矩阵。

## 仓库维护约定

- 每次代码、配置或实验记录有实质更新，都要更新 `CHANGELOG.md`。
- 重要结论同步写进 `README.md` 或 `reference/EXPERIMENT_LOG.md`。
- 不提交 SSH 私钥、SSH 登录说明、模型权重、checkpoint、训练输出大文件。
- 第三方框架源码只作为本地参考，默认不提交到本仓库；仓库内只记录来源链接和我们自己的适配说明。

