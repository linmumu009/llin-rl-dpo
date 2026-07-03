# 训练框架评估矩阵

日期：2026-07-03

| 框架 | 当前适配判断 | 能否跑通 | 效率评估 | 效果评估 | 当前风险 |
|---|---|---:|---|---|---|
| ModelScope ms-swift | 阿里 ModelScope 框架；官方文档覆盖 Qwen3.6/Qwen3.5、DPO、人类对齐、Ascend NPU、FSDP/FSDP2/DeepSpeed/Megatron | 已跑通：Qwen3.6-27B + DPO + LoRA + FSDP2 + 8 NPU 完成 1 step 和 20 step；FSDP2 checkpoint 保存通过，sharded resume 仍失败；`FULL_STATE_DICT + save_only_model` 已导出普通 LoRA adapter，并完成 adapter 重新加载和固定 prompts 对照推理 | 20 step 合成数据：平均约 5.92s/step；保存版 20 step：平均约 4.89s/step，samples/s 1.636，显存记录 51.85 GiB；non-thinking 128 token 固定 prompts 推理 base `4.7594 tokens/s`，adapter `3.0811 tokens/s` | 合成数据 loss 下降到 train_loss=0.0722，未做真实验证集；adapter safetensors 可读取，包含 992 个 LoRA tensor；2 step 合成 adapter 与 base 输出方向基本一致，暂不能说明效果变化；`enable_thinking=false` 仍保留空 `<think></think>` 前缀 | sharded resume 先后遇到 `assign` 签名和缺视觉塔完整权重问题；真实数据、AICore 利用率、`chunk_gated_delta_rule` warning 影响和 deepspeed 对照仍待测；128 token 下 base 输出仍可能截断 |
| LLaMA-Factory NPU | 官方文档覆盖 Atlas A2/A3 NPU、torch_npu、Qwen3.6、DPO；生态成熟 | 待测试 | 待测 | 待测 | Qwen3.6 虽在模型表中，但 Ascend 上的 `qwen3_5` patch 和 27B DPO 仍需实测 |
| MindSpeed-LLM FSDP2/DPO | 服务器已有 MindSpeed-LLM 镜像；当前 `llin-rl-dpo` 使用该镜像作为 8 NPU 安全底座 | 8 NPU + HCCL smoke test 通过；作为 ms-swift 运行底座已支撑 Qwen3.6 DPO 1 step | 待测 | 待测 | MindSpeed-LLM 原生命令仍未直接验证 `qwen3_5` DPO；当前成功路径来自 ms-swift |
| MindSpeed-RL DPO | 服务器已有官方 RL 镜像；官方支持 DPO，但样例是 Qwen3-30B-A3B | 容器底座曾卡在 NPU 可见性；当前 8 NPU 底座在 MindSpeed-LLM 镜像通过 | 暂不可测 | 暂不可测 | 目标 qwen3.6-27B 是 `qwen3_5` conditional/multimodal 配置，不能直接套 Qwen3-30B-A3B DPO 样例 |
| vLLM Ascend | 服务器已有 qwen3.6-27B 推理镜像和 compose；官方支持 Qwen3.6-27B 推理 | 推理路径已有参考，训练不适用 | 可作为推理效率/效果评估 | 可做训练前后评测服务 | vLLM 是推理框架，不能直接做 DPO 训练 |
| verl Ascend / FSDP | 官方 Ascend tutorial 存在；支持表列出 Qwen3、Qwen3.5、Qwen3-Next 等 RL 组合 | 未测试 | 待测 FSDP2/FSDP 吞吐、显存占用 | 待测 | 看到 Qwen3.5-27B GRPO 证据，但未看到 Qwen3.6-27B DPO 直接证据 |
| FlagScale / FlagOS / VeRL-FL | 多芯片统一训练/RL/推理工具链；支持列表含 Qwen2/2.5/3，RL 插件对应 VeRL-FL | 未测试 | 待测 | 待测 | Qwen3.6 + DPO 直接证据不足，首轮不优先 |
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

1. ModelScope ms-swift：优先，因为官方同时覆盖 Qwen3.6、DPO、Ascend NPU 和多种分布式后端。
2. LLaMA-Factory NPU：第二优先，因为官方同时覆盖 NPU、Qwen3.6、DPO，适合快速 LoRA/DPO 试验。
3. MindSpeed-RL / MindSpeed-LLM：Ascend 原生路线，适合继续做深度适配，但不要作为唯一主线。
4. verl Ascend：作为 GRPO/RL 框架候选保留。
5. torch_npu 原生 FSDP：兜底路线，适合验证模型加载和最小 DPO 循环，但工程成本最高。
