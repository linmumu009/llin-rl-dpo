# Ascend + Qwen3.6 + DPO 框架调查

日期：2026-07-03

目标不是只找“能训练 LLM”的框架，而是筛出在我们当前条件下最可能落地的路线：

- 硬件：华为 Ascend A3 训练服务器，当前容器内 8 逻辑 NPU 和 HCCL 已验证。
- 模型：`/data/models/Qwen3.6-27B`，本地 `config.json` 为 `model_type=qwen3_5`、`architectures=Qwen3_5ForConditionalGeneration`。
- 任务：DPO，后续也可能扩展到 GRPO/PPO/RLHF。
- 边界：只使用我们自己的 `llin-rl-dpo` 容器，不改动物理机，不碰别人的容器。

## 结论排序

| 优先级 | 框架 | 厂商/生态 | 为什么进入候选 | 当前判断 |
|---:|---|---|---|---|
| 1 | ModelScope ms-swift | 阿里 / ModelScope | 官方 README 明确列出 Qwen3.6、DPO、人类对齐、Ascend NPU、FSDP/FSDP2/DeepSpeed/Megatron；NPU 最佳实践还给出 Ascend 环境、DPO 支持、Qwen3.5 FLA patch 路线 | 第一优先级，建议下一个实测 |
| 2 | LLaMA-Factory NPU | LLaMA-Factory 社区 | 官方文档有 NPU 安装/训练页，README 支持 Qwen3.6 和 DPO；生态成熟，适合快速验证 LoRA/DPO | 第二优先级，需实测 `qwen3_5` 和 NPU patch |
| 3 | MindSpeed-RL | 华为 Ascend | 原生 Ascend RL 框架，官方支持 DPO；但 DPO 示例目前偏 Qwen3-30B-A3B，不是 qwen3_5/Qwen3.6-27B | 仍保留，但不是唯一主线 |
| 4 | verl Ascend | verl + Ascend | 官方 Ascend tutorial 存在，模型/算法表列出大量 Qwen 系列 RL 组合，包括 Qwen3.5-27B GRPO 和 Qwen3-Next | 适合 RL/GRPO 路线；DPO/Qwen3.6-27B 直接证据不足 |
| 5 | FlagScale / FlagOS / VeRL-FL | 智源 FlagOS | 统一多芯片训练/RL/推理工具链，声明训练、强化学习、推理插件化跨芯片；支持列表含 Qwen2/2.5/3 | 中长期候选；Qwen3.6 + DPO 直接证据不足 |
| 6 | vLLM Ascend | vLLM Ascend | 官方支持 Qwen3.6-27B 推理，服务器已有可参考 compose | 只作推理/评测服务，不作为 DPO 训练框架 |

## 关键证据

### ModelScope ms-swift

官方仓库说明：

- `ms-swift` 是 ModelScope 社区的大模型微调与部署框架。
- README 明确列出支持 `Qwen3.6`、`Qwen3.5`、`Qwen3` 等模型。
- 支持训练任务包括 `DPO`、`KTO`、`RM`、`CPO`、`SimPO`、`ORPO`。
- 硬件支持列表包含 Ascend NPU。
- 分布式支持包含 DDP、DeepSpeed ZeRO2/3、FSDP/FSDP2、Megatron。

官方 NPU 最佳实践说明：

- 增加了对昇腾 NPU 的支持，可在 NPU 上做模型微调和推理。
- 推荐基础版本包括 Python 3.10/3.11、CANN、torch、torch_npu。
- 支持范围速览里列出 DPO、DDP、FSDP、FSDP2、DeepSpeed、MindSpeed(Megatron) 已支持。
- 已验证 RL 组合包含 DPO + Qwen2.5/Qwen3-8B + DeepSpeed + vLLM-Ascend。
- Qwen3.5 最佳实践存在单独页面，并有 `Qwen3.5 FLA补丁说明`。我们的 Qwen3.6-27B 配置落在 `qwen3_5` 系列，优先从这里切入更合理。

来源：

- https://github.com/modelscope/ms-swift
- https://swift.readthedocs.io/zh-cn/latest/BestPractices/NPU-support.html
- https://swift.readthedocs.io/zh-cn/latest/BestPractices/Qwen3_5-Best-Practice.html
- https://swift.readthedocs.io/zh-cn/latest/Instruction/RLHF.html

初步实测计划：

1. 在 `reference/` 拉取 `ms-swift` 源码作为本地参考，不提交第三方源码。
2. 在当前 `llin-rl-dpo` 容器中先只做 import/version/help 级别 smoke test。
3. 检查 `ms-swift` 是否识别本地 `/models/Qwen3.6-27B` 的 `qwen3_5` 配置。
4. 若模型加载通过，再准备最小 DPO 数据集，跑 1 个 optimizer step。

### LLaMA-Factory NPU

官方文档说明：

- LLaMA-Factory 支持 Qwen3.6，模型表中给出 `Qwen3.6 27B/35B qwen3_6`。
- 支持 DPO、PPO、KTO、ORPO、SimPO 等训练方式。
- NPU 文档面向 Atlas A2/A3 训练系列设备，依赖 HDK、CANN、torch_npu。
- 官方提供 NPU 镜像，包括 `hiyouga/llamafactory:latest-npu-a3` 和 `quay.io/ascend/llamafactory:latest-npu-a3`。
- NPU 训练文档说明单机多卡可通过 `ASCEND_RT_VISIBLE_DEVICES` 启动。

来源：

- https://github.com/hiyouga/LLaMA-Factory
- https://llamafactory.readthedocs.io/zh-cn/latest/multibackend/npu/npu_installation.html
- https://llamafactory.readthedocs.io/zh-cn/latest/multibackend/npu/npu_training.html

初步实测计划：

1. 先不新建额外容器，优先在当前容器里拉源码读模型注册和 NPU requirements。
2. 如果必须换官方 NPU 镜像，也只重建我们自己的 `llin-rl-dpo` 容器。
3. 先测 `llamafactory-cli version`、NPU 可见、Qwen3.6 config 加载。
4. 再跑 1 step DPO。

### MindSpeed-RL

官方仓库说明：

- MindSpeed-RL 是基于昇腾生态的强化学习加速框架。
- 官方 DPO 入口存在，支持模型列表里 DPO 示例是 `Qwen3-30B-A3B`。
- 仓库页面提示项目已迁移至 GitCode。
- 当前我们已验证 MindSpeed-LLM 镜像里的 8 NPU 和 HCCL 通信，但 MindSpeed-LLM FSDP2 对 `qwen3_5` 没有直接模型类型支持。

来源：

- https://gitee.com/ascend/MindSpeed-RL
- https://gitcode.com/Ascend/MindSpeed-RL

下一步：

- 不放弃，但不要再把它当作唯一主线。
- 若 ms-swift 能复用 MindSpeed/Megatron 路径，可能比直接改 MindSpeed-RL 更快。

### verl Ascend

官方文档说明：

- `verl` 是 LLM post-training / RL 训练框架，集成 PyTorch FSDP、Megatron-LM、vLLM、SGLang 等。
- 官方 Ascend tutorial 明确表示昇腾支持 verl 使用和开发。
- Ascend 模型/算法支持表列出 Qwen3、Qwen3.5、Qwen3-Next 等多种 RL 组合。
- 已列出 Qwen3.5-27B GRPO + FSDP2 + vLLM，但没有看到 Qwen3.6-27B DPO 的直接条目。

来源：

- https://github.com/verl-project/verl
- https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/README.md
- https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/model_support/model_and_algorithm_support.md

下一步：

- 作为 GRPO/RL 框架候选保留。
- 若用户后续改成 GRPO 或在线 RL，verl Ascend 值得单独开一轮。

### FlagScale / FlagOS / VeRL-FL

官方仓库说明：

- FlagScale 是 FlagOS 的核心组件，定位为统一开源 AI 系统软件栈。
- 目标是跨多种芯片迁移训练、强化学习和推理。
- 插件映射里包括训练 `Megatron-LM-FL`、RL `VeRL-FL`、推理 `vllm-plugin-FL`。
- 支持列表包含 Qwen2/2.5/3 训练和 Qwen3 推理，但未看到 Qwen3.6-27B + DPO 的直接证据。

来源：

- https://github.com/flagos-ai/FlagScale

下一步：

- 暂列中长期候选。
- 不作为第一轮实验，除非 ms-swift/LLaMA-Factory 都卡住。

### vLLM Ascend

当前定位：

- 服务器已有 Qwen3.6-27B vLLM Ascend 推理 compose。
- 官方 vLLM Ascend 文档支持 Qwen3.6-27B 推理。
- 它不能直接做 DPO 训练，但可以作为训练前后效果评测服务，尤其用于固定 prompt 集的生成对比。

来源：

- https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html

## 下一步执行顺序

1. `ms-swift`：拉源码到本地 `reference/ms-swift`，查 `qwen3_5`、`qwen3_6`、NPU patch、DPO 示例。
2. `ms-swift`：在 `llin-rl-dpo` 容器中做最小 import/help/config-load smoke test。
3. 如果 `ms-swift` 卡在依赖或模型加载，再切 `LLaMA-Factory NPU`。
4. 如果两个轻量框架都卡，再回到 MindSpeed-RL/MindSpeed-LLM 做模型类型适配。
5. 每完成一个实测阶段，更新本调查、评估矩阵、实验日志和 CHANGELOG。
