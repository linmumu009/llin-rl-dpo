# 训练框架评估矩阵

日期：2026-07-03

| 框架 | 当前适配判断 | 能否跑通 | 效率评估 | 效果评估 | 当前风险 |
|---|---|---:|---|---|---|
| MindSpeed-RL DPO | 服务器已有官方 RL 镜像；官方支持 DPO，但样例是 Qwen3-30B-A3B | smoke test 未通过 | 暂不可测 | 暂不可测 | torch_npu 可 import，但 NPU device_count 为 0；目标 qwen3.6-27B 是 `qwen3_5` conditional/multimodal 配置 |
| MindSpeed-LLM FSDP2/DPO | 服务器已有 MindSpeed-LLM 镜像；当前 `llin-rl-dpo` 使用该镜像作为 8 NPU 安全底座 | 8 NPU + HCCL smoke test 通过；qwen3.6 模型加载未通过 | 待测 | 待测 | FSDP2 支持 DPO，但 Transformers/MindSpeed 当前不直接识别 `qwen3_5` |
| vLLM Ascend | 服务器已有 qwen3.6-27B 推理镜像和 compose；官方支持 Qwen3.6-27B 推理 | 推理路径已有参考，训练不适用 | 可作为推理效率/效果评估 | 可做训练前后评测服务 | vLLM 是推理框架，不能直接做 DPO 训练 |
| verl NPU / FSDP | MindSpeed-RL 仓库包含 `verl_npu` 适配和 FSDP 测试脚本 | 未测试 | 待测 FSDP2/FSDP 吞吐、显存占用 | 待测 | 对 qwen3_5 / vision conditional generation 的支持未知 |
| torch_npu FSDP 原生路线 | 最通用，但需要自己搭 DPO 训练循环或接 TRL/verl | 未测试 | 待测 | 待测 | 工程量更大；Transformers qwen3_5 支持版本要求高 |

## 统一评估指标

跑通：

- 容器环境 import 成功。
- NPU 可见。
- 数据预处理成功。
- 权重转换或模型加载成功。
- 训练至少完成 1 个 optimizer step。
- checkpoint 可保存，必要时可恢复。

效率：

- tokens/s 或 samples/s。
- step time P50/P95。
- HBM 峰值和均值。
- AICore 利用率。
- HCCL/通信等待占比。
- 启动耗时、数据预处理耗时、权重转换耗时。

效果：

- DPO loss 曲线。
- chosen logprob / rejected logprob 差值。
- reward margin 或 preference accuracy。
- 固定验证集上的 win rate。
- 少量人工样例检查。

## 当前优先级

1. MindSpeed-RL：优先，因为服务器已有专用镜像和官方 DPO 入口。
2. verl NPU/FSDP：如果 MindSpeed-RL 卡在 qwen3_5 架构或 mcore 转换，转向这个路线。
3. torch_npu 原生 FSDP：作为兜底路线，适合验证模型加载和最小 DPO 循环，但工程成本最高。
