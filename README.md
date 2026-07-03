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

不要只把 MindSpeed-RL 当作唯一主线。2026-07-03 的外部调查显示，阿里 ModelScope `ms-swift` 和 LLaMA-Factory NPU 都已经显式覆盖 Ascend NPU、DPO 和 Qwen3.6 相关能力，其中 `ms-swift` 还提供 NPU 最佳实践、Qwen3.5 最佳实践和 MindSpeed/Megatron-SWIFT 路线，因此下一轮优先实测 `ms-swift`。

完整调查见 [reference/ASCEND_QWEN36_FRAMEWORK_SURVEY.md](reference/ASCEND_QWEN36_FRAMEWORK_SURVEY.md)。

但当前 `qwen3.6-27B` 的 `config.json` 显示它不是普通 Qwen3 dense 模型，而是：

- `architectures`: `Qwen3_5ForConditionalGeneration`
- `model_type`: `qwen3_5`
- `language_model_only`: `false`
- 包含 `text_config` 和 `vision_config`

所以不能把 MindSpeed-RL 的 `qwen3_30b_a3b` DPO 样例直接照抄到 qwen3.6-27B。第一阶段必须先做环境和架构支持 smoke test。

## 评估框架

详见 [reference/FRAMEWORK_EVAL_MATRIX.md](reference/FRAMEWORK_EVAL_MATRIX.md)。

当前优先级：

1. ModelScope ms-swift
2. LLaMA-Factory NPU
3. MindSpeed-RL / MindSpeed-LLM
4. verl Ascend / FSDP
5. torch_npu 原生 FSDP

## 版本说明

版本记录见 [CHANGELOG.md](CHANGELOG.md)。

当前版本：

- `v0.1.4`：扩展框架调查到阿里 ModelScope `ms-swift`、LLaMA-Factory NPU、verl Ascend、FlagScale/FlagOS、vLLM Ascend；把下一轮优先级调整为先实测 `ms-swift`。
- `v0.1.5`：完成 `ms-swift` 首轮实测；Transformers 5.12 已能识别并 meta 构建 Qwen3.6/Qwen3_5，ms-swift 已能识别本地模型并加载 processor；当前阻塞变为 ms-swift 默认 NPU model patch 与当前 MindSpeed/Triton/CANN 版本不兼容。

## 仓库维护约定

- 每次代码、配置或实验记录有实质更新，都要更新 `CHANGELOG.md`。
- 重要结论同步写进 `README.md` 或 `reference/EXPERIMENT_LOG.md`。
- 不提交 SSH 私钥、SSH 登录说明、模型权重、checkpoint、训练输出大文件。
- 第三方框架源码只作为本地参考，默认不提交到本仓库；仓库内只记录来源链接和我们自己的适配说明。

## 当前容器状态

`llin-rl-dpo` 容器已创建并运行，当前使用设备：

- `ASCEND_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`

已验证：

- `torch_npu.npu.device_count()` 返回 `8`
- `scripts/npu_multicard_smoke.py` 逐卡 tensor 计算全部返回 `4.0`
- `scripts/hccl_smoke.py` 8 进程 HCCL all-reduce 返回 `36.0`

关键修复：

- Ascend Docker Runtime 按 `ASCEND_VISIBLE_DEVICES` 做设备挂载、cgroup 配置和 CANN runtime library 注入。
- 只设置 `ASCEND_RT_VISIBLE_DEVICES` 或手动 `--device=/dev/davinci*` 不足以让 torch_npu 正常看到设备。
- 设备 `0` 已被已有容器占用；使用 `ASCEND_VISIBLE_DEVICES=0` 时容器内 `npu-smi` 报 `device is used`。

## 当前训练阻塞点

qwen3.6-27B 的本地配置是：

- `model_type`: `qwen3_5`
- `architectures`: `Qwen3_5ForConditionalGeneration`
- `task`: `image-text-to-text`

当前训练容器内：

- Transformers `4.57.1` 不识别 `qwen3_5`，`AutoConfig.from_pretrained("/models/Qwen3.6-27B")` 会失败。
- MindSpeed-LLM FSDP2 支持 `qwen3`、`qwen3-moe`、`qwen3-next` 等，但未直接支持 `qwen3_5`。
- vLLM Ascend 官方文档支持 Qwen3.6-27B 推理，但这不能直接等价为 DPO 训练支持。

## 服务器网络约束

服务器只能访问部分中国大陆网站。后续服务器侧依赖默认使用大陆镜像、ModelScope、GitCode/Gitee；GitHub/Hugging Face 资料优先在本地获取后通过 `scp` 同步到服务器我们的工作区。

## ms-swift 首轮实测结论

`ms-swift` 是目前对 Qwen3.6-27B 支持证据最强的训练框架。

已通过：

- `transformers==5.12.1` 可识别 `model_type=qwen3_5`
- 可在 meta device 上构建 `Qwen3_5ForConditionalGeneration`
- `ms-swift` 可识别本地 `/models/Qwen3.6-27B` 为 `qwen3_5`
- 关闭 NPU model patch 后，processor/tokenizer/chat_template 加载通过

未通过：

- 默认开启 NPU model patch 时，Qwen3.5/Qwen3.6 linear attention 的 MindSpeed Triton 路径编译失败。
- 当前容器版本组合是 `torch_npu 2.7.1.post4`、`mindspeed 0.12.1`、`triton 3.2.0`、CANN 9.0.0，和 ms-swift 文档中的 Qwen3.5 NPU patch 验证组合不一致。
