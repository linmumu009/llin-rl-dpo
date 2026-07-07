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

- `v0.1.20`：按“我们旧 MindSpeed-MM + transformers 5.2.0 + torch_npu 2.7.1.post4”做完整老板数据、8 卡全参、`cutoff=4096`、`micro_batch_size=2`、activation offload、10 step 对照；因旧源码不支持 `openai/qwen3_6`，数据转为 `sharegpt`、模板用 `qwen3_vl`。结果第 2 step 未复现 rotary 561002，跑过 6 step 后在视觉塔 forward OOM；说明旧版本栈没有老板的第二步 rotary 错，但该对照仍受模板/预处理差异影响。
- `v0.1.19`：补充 561002 的变量对照和根因收敛：`cutoff=2048` 通过，`micro_batch_size=1` 仍在 row20 形状复现，关闭 activation offload 仍复现，`ASCEND_LAUNCH_BLOCKING=1` 确认真实触发点是 `npu_rotary_mul_backward/aclnnRotaryPositionEmbeddingGrad`；旧 MindSpeed-MM/transformers 5.2 对照通过是因为模板把 row20 从 `2506/2052` 缩到 `475/21`，不是证明底层算子问题消失。
- `v0.1.18`：按老板给出的真实数据、配置、启动脚本和 MindSpeed-MM 源码快照，在我们自己的 `llin-rl-dpo` 容器内原生复现 `aclnnRotaryPositionEmbeddingGrad error 561002`；全量数据第 2 step rank4 失败，最小化到前 32 条数据跑 2 step 仍可复现，16-31 条单独 1 step 不复现。详见 [reference/RJX_561002_REPRO_20260707.md](reference/RJX_561002_REPRO_20260707.md)。
- `v0.1.17`：补充冻结视觉塔 3-step 对照；同样 8 NPU、`cutoff=4096`、long-answer、原始上下文超过 4096，`freeze: ['model.visual']` 完成 3 step 并保存 checkpoint。与全参数 OOM 形成直接对照，支持纯文本 SFT/DPO 默认冻结视觉塔。
- `v0.1.16`：复查老板反馈的 8 卡全参 `cutoff=4096`、原始上下文超过 4096、3 step 场景；第 1 step 跑通，随后在第二步 backward / FSDP reduce-scatter 附近 OOM，报 `runtime result = 207001`，working operator 显示 `aclnnFlashAttentionScore`；未出现 rotary `561002` 或 `aclnnCat`。
- `v0.1.15`：复查 MindSpeed-MM `cutoff=4096` 的 2 step 与 long-answer 场景；确认 long-answer cache 实际为 4096 token，监督 label 为 4073 token、覆盖到位置 4095，训练仍在 8 NPU 上通过；同时记录显存 max reserved 约 61.1GB，说明老板截图中的 torch 原生 rotary patch / `aclnnCat` OOM 仍是合理风险。
- `v0.1.14`：补充 MindSpeed-MM Qwen3.6-27B SFT cutoff 复现实验；在 8 NPU 隔离 venv 路径下 `cutoff=2048` 和 `cutoff=4096` 均完成 1 step，未复现 `aclnnRotaryPositionEmbeddingGrad error 561002`；同时记录 4 卡 OOM、transformers 版本不匹配和容器 NPU 映射差异。
- `v0.1.13`：新增 512 条训练运维/框架评估半真实 DPO 数据，完成 `ms-swift + Qwen3.6-27B + DPO + LoRA + FSDP2` 的 100 step 试跑；adapter 导出和重新加载推理均通过，但固定 prompts 上 base/adapter 输出完全一致，暂不能证明泛化效果。
- `v0.1.12`：固定 prompts 对照支持 `ENABLE_THINKING=false` 并完成 128 token 实测；确认 ms-swift 会关闭思考内容但仍保留空的 `<think></think>` 前缀。base `4.7594 tokens/s`，adapter `3.0811 tokens/s`。
- `v0.1.11`：新增固定 prompts 的 base/adapter 对照评测脚本；实测 base `4.5922 tokens/s`、adapter `3.0373 tokens/s`，两边输出方向基本一致，2 step 合成 DPO adapter 暂无可解释效果差异。
- `v0.1.10`：新增 adapter 非交互推理 smoke test，确认 `FULL_STATE_DICT + save_only_model` 导出的 LoRA adapter 可被 ms-swift 重新加载并生成；16 token smoke 输出被截断，只证明加载/推理链路，不代表效果。
- `v0.1.9`：新增可选 `SwiftModel.load_state_dict(assign=...)` runtime patch，确认 resume 第一阶段错误可绕过；进一步定位到 FSDP2 sharded checkpoint 缺完整视觉塔权重。同时新增 `FULL_STATE_DICT + save_only_model` 配置，成功导出普通 LoRA adapter。
- `v0.1.8`：训练脚本支持 checkpoint/resume 参数；20 step 保存测试成功生成 `checkpoint-10` 和 `checkpoint-20`，但从 FSDP2 checkpoint 恢复失败，当前恢复链路未通过。
- `v0.1.7`：新增合成 DPO 数据生成脚本，完成 256 条合成数据上的 20 step 稳定性测试；运行成功，平均约 `5.92s/step`，显存记录约 `51.93 GiB`。
- `v0.1.6`：`ms-swift + Qwen3.6-27B + DPO + LoRA + FSDP2` 在 8 NPU 上完成 1 个 optimizer step；记录 tiny DPO 数据、可复用启动脚本、环境版本和初步效率指标。
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

## 当前训练状态

qwen3.6-27B 的本地配置是：

- `model_type`: `qwen3_5`
- `architectures`: `Qwen3_5ForConditionalGeneration`
- `task`: `image-text-to-text`

## MindSpeed-MM SFT cutoff 复现

针对老板反馈的 MindSpeed-MM SFT `cutoff > 2048` 可能触发 `aclnnRotaryPositionEmbeddingGrad error 561002`，已完成一轮可复核复现。

### 同事真实配置复现 561002

2026-07-07 按老板给出的同事环境信息复制数据、配置、启动脚本和日志到我们自己的 `llin-rl-dpo` 工作区，并复制 `mindspeed_mm_rjx` 容器中的 MindSpeed-MM 源码快照到 `reference/MindSpeed-MM-rjx-snapshot`。我们没有修改同事容器、没有改宿主机，也没有做 rotary patch。

结论：可以复现。

- 全量 `20260702_openai.jsonl`、8 NPU、Full SFT、`cutoff_len=4096`、`micro_batch_size=2`、`enable_activation_offload=true`：iteration 1 通过，iteration 2 rank4 在 `aclnnRotaryPositionEmbeddingGrad` 报 `561002`，`reserveAlignNum = 2592 too large`。
- 前 32 条数据、同配置、2 step：同样 iteration 1 通过，iteration 2 rank4 复现同一个 `561002`。
- 只取原全量第 16-31 条数据、同配置、1 step：通过，说明不是单独这 16 条数据一出现就失败，而是和前一轮之后第二个 global batch 的 rank-local shape/状态有关。
- 失败 rank4 的第二步本地样本来自原始行 `20` 和 `28`，tokenized 长度分别为 `2506` 和 `1264`；全量数据中只有两条达到 `4096`，所以它不是“任意 4096-token 样本都会失败”的简单问题。
- 变量对照显示：`cutoff=2048` 通过；`micro_batch_size=1` 仍在原始 row20 形状处复现；关闭 activation offload 仍复现；`ASCEND_LAUNCH_BLOCKING=1` 确认真实失败点是 `npu_rotary_mul_backward`。
- 旧 MindSpeed-MM/`transformers 5.2.0` 路径用老板数据内容可跑通，但不是等价配置：旧源码不支持 `formatting: openai` 和 `template: qwen3_6`，转换到 `sharegpt + qwen3_vl` 后 row20 从 `2506/2052` 变成 `475/21`，避开了危险 shape。
- 更严格的完整数据旧版本栈对照中，8 卡全参、`cutoff=4096`、`micro_batch_size=2`、activation offload、10 step 配置下第 2 step 没有 rotary 561002，完成 6 step 后在旧 `qwen3_vl` 视觉塔 forward OOM。这说明老板第二步 rotary 错在旧版本/旧模板路径下不复现，但不能把原因归结为单一 Python 包版本，因为旧源码必须换数据格式和模板。

详细路径、日志和最小复现记录见 [reference/RJX_561002_REPRO_20260707.md](reference/RJX_561002_REPRO_20260707.md)。

关键环境：

- 框架源码：`reference/MindSpeed-MM`，Qwen3.6 示例使用 `mindspeed_mm/fsdp/models/qwen3_5`。
- 配套 MindSpeed：`reference/MindSpeed-26.0.0_core_r0.12.1`。
- 模型：`/models/Qwen3.6-27B`。
- 权重格式：HF 权重已转换为 DCP，路径 `/workspace/llin-rl-dpo/checkpoints/msmm-qwen36-27b-dcp`。
- 运行容器：复用我们自己的 `llin-rl-dpo` 容器，新建隔离 venv `/workspace/llin-rl-dpo/.venvs/msmm-qwen36`，没有改宿主机和别人的容器。
- 设备：`ASCEND_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`，8 张逻辑 NPU。
- 关键 Python 版本：`torch 2.7.1+cpu`，`torch_npu 2.7.1.post4`，`transformers 5.2.0`，`accelerate 1.2.0`，`datasets 5.0.0`，`triton-ascend 3.2.1`。

结果：

- `cutoff=2048`：8 NPU 完成 1 step，exit code `0`；`elapsed time per iteration (ms): 125604.0`，`global batch size: 8`，`loss: 1.220408E+01`，未出现 `RotaryPositionEmbeddingGrad` 或 `561002`。
- `cutoff=4096`：8 NPU 完成 1 step，exit code `0`；`elapsed time per iteration (ms): 53971.2`，`global batch size: 8`，`loss: 1.006310E+01`，未出现 `RotaryPositionEmbeddingGrad` 或 `561002`。
- `cutoff=4096` 短 answer：8 NPU 完成 2 step，exit code `0`；iteration 1/2 分别约 `16419.2 ms` / `14716.3 ms`，未出现 `RotaryPositionEmbeddingGrad`、`561002` 或 OOM。
- `cutoff=4096` long-answer：8 NPU 完成 2 step，exit code `0`；iteration 1/2 分别约 `16957.6 ms` / `14105.7 ms`；cache 中 `input_ids` 全部为 `4096`，`labels != -100` 为 `4073`，覆盖位置 `23..4095`；未出现 `aclnnCat` OOM、`RotaryPositionEmbeddingGrad` 或 `561002`。
- `cutoff=4096` long-answer 全参数：8 NPU 目标 3 step，exit code `1`；配置确认 `freeze: []`，iteration 1/3 跑通，`17755.8 ms`，loss `1.215048E+01`；随后 OOM，`runtime result = 207001`，working operator 显示 `aclnnFlashAttentionScore`，当时部分 rank 只剩约 `29-97 MiB` 可用显存。
- `cutoff=4096` long-answer 冻结视觉塔：8 NPU 完成 3 step，exit code `0`；配置确认 `freeze: ['model.visual']`，iteration 1/2/3 分别约 `16885.2 ms` / `13886.2 ms` / `19824.2 ms`；checkpoint 保存到 `outputs/msmm-qwen36-sft-cutoff4096-longanswer-freezevisual-3step/iter_0000003`。

对老板截图中 OOM 的解释：

- 我们本轮确认不是“训练数据没到 4096”。long-answer cache 已证明真实训练 token 到 4096，且几乎全长都有 answer loss。
- 冻结视觉塔时，`cutoff=4096` long-answer 2 step 可以跑通，但 max reserved memory 已约 `61132 MB`，距离 64GB HBM 很近。
- 冻结视觉塔 3 step 已通过；去掉 `freeze: model.visual` 做全参数后，8 卡只能完成第 1 step，第二步附近 OOM。这说明纯文本任务默认冻结视觉塔是合理的，也说明老板说的 8 卡全参 4096 显存不够是对的。
- 当前我们复现到的全参错误是显存硬限制，报 `207001`，不是 rotary `561002`；如果再叠加 torch 原生 rotary patch 的额外 `torch.cat` 分配，`aclnnCat` OOM 也很合理。

排查中遇到的真实限制：

- 4 NPU 路径可以启动但 Qwen3.6-27B SFT 初始化 OOM，官方 MindSpeed-MM Qwen3.6 27B 脚本默认是 `NPUS_PER_NODE=16`。
- `transformers 5.13.0` 会触发 `create_causal_mask() got an unexpected keyword argument 'cache_position'`；按 MindSpeed-MM Qwen3.6 文档降到 `transformers==5.2.0` 后通过。
- 新建多个 MindSpeed-MM 容器时出现 `torch.npu.device_count()==0`，即使设备节点存在；目前稳定可复核路径是在已有我们自己的 `llin-rl-dpo` 容器内用隔离 venv 跑 MindSpeed-MM。

当前已经跑通：

- 框架：`ms-swift 4.5.0.dev0`
- 路线：DPO + LoRA + Transformers/FSDP2
- 设备：8 逻辑 NPU
- 模型：本地 `/models/Qwen3.6-27B`
- 数据：`datasets/tiny_dpo.jsonl`
- 脚本：`scripts/run_ms_swift_qwen36_dpo_smoke.sh`
- 结果：`global_step/max_steps=1/1`
- DPO loss：`0.69140625`
- 1 step 用时：约 `139.6s`
- 记录显存：约 `51.19 GiB`

已完成进一步稳定性测试：

- 数据：`datasets/synthetic_dpo_256.jsonl`，由 `scripts/make_synthetic_dpo.py` 生成。
- 步数：20 optimizer steps。
- 输出目录：`/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-stability-20step/v0-20260703-120434`
- 结果：`global_step/max_steps=20/20`
- 总训练时长：`118.3962s`
- 平均速度：约 `5.918s/step`
- samples/s：`1.351`
- 记录显存：约 `51.93 GiB`
- 训练 loss：`0.0722111`

已完成 checkpoint 测试：

- 脚本：`scripts/run_ms_swift_qwen36_dpo_smoke.sh` 已支持 `SAVE_STRATEGY`、`SAVE_STEPS`、`SAVE_TOTAL_LIMIT`、`RESUME_FROM_CHECKPOINT`。
- 保存测试：20 step 成功，`save_steps=10`，输出 `checkpoint-10` 和 `checkpoint-20`。
- checkpoint 形态：FSDP2 sharded checkpoint，包含 `pytorch_model_fsdp_0`、`optimizer_0`、scheduler、trainer state 和每 rank RNG。
- checkpoint 大小：`checkpoint-10` 约 `713M`，`checkpoint-20` 约 `713M`。
- 恢复测试：从 `checkpoint-10` 恢复到 `max_steps=12` 失败。
- 失败点：`SwiftModel.load_state_dict() got an unexpected keyword argument 'assign'`。

已完成 resume/导出进一步排查：

- 新增 `patches/sitecustomize.py`，通过 `LLIN_SWIFTMODEL_ASSIGN_PATCH=1` 可选启用 `SwiftModel.load_state_dict(assign=...)` 兼容补丁。
- 启用补丁后，resume 不再卡在 `assign`，但继续失败于 `Missing key in checkpoint state_dict: model.model.visual.patch_embed.proj.weight`。
- 判断：默认 FSDP2 sharded checkpoint 保存的是 adapter/trainable 状态，恢复时 loader 按完整模型状态查找，二者不匹配。
- 新增 `configs/fsdp2_full_state.json`，使用 `FULL_STATE_DICT`。
- 使用 `FULL_STATE_DICT + SAVE_ONLY_MODEL=true` 跑 2 step，成功导出普通 LoRA adapter。
- adapter 产物：`adapter_model.safetensors` 约 `223M`，`adapter_config.json` 指向 `/models/Qwen3.6-27B`，safetensors 可读取，包含 `992` 个 tensor。

已完成 adapter 重新加载推理测试：

- 新增脚本：`scripts/run_ms_swift_adapter_infer_smoke.sh`。
- 新增数据：`datasets/tiny_infer.jsonl`。
- 加载 adapter：`/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step/v0-20260703-130918/checkpoint-2`。
- 命令：`swift infer --model /models/Qwen3.6-27B --model_type qwen3_5 --adapters <checkpoint-2> --infer_backend pt --device_map auto`。
- 结果：exit code `0`，生成文件 `/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step/v0-20260703-130918/checkpoint-2/infer_result/20260703-132529.jsonl`。
- 指标：`37` prompt tokens，`16` generated tokens，runtime `49.2921s`，tokens/s `0.3246`。
- 注意：`MAX_NEW_TOKENS=16` 的 smoke 输出被截断，只说明 adapter 加载和推理链路可用，不代表效果。

已完成固定 prompts 的 base/adapter 对照：

- 新增脚本：`scripts/run_ms_swift_fixed_prompt_eval.sh` 和 `scripts/run_ms_swift_base_adapter_compare.sh`。
- 新增数据：`datasets/fixed_eval_prompts.jsonl`。
- base 结果：`/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/base-20260703.jsonl`。
- adapter 结果：`/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/adapter-20260703.jsonl`。
- base：`128` generated tokens，runtime `27.8733s`，tokens/s `4.5922`。
- adapter：`128` generated tokens，runtime `42.1431s`，tokens/s `3.0373`。
- 判断：2 step 合成 DPO adapter 与 base 输出方向基本一致，暂不能说明效果变化；adapter 推理有可观开销。

已完成 non-thinking 128 token 对照：

- 脚本参数：`RUN_ID=nonthinking-128-20260703 MAX_NEW_TOKENS=128 ENABLE_THINKING=false scripts/run_ms_swift_base_adapter_compare.sh`。
- base 结果：`/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/base-nonthinking-128-20260703.jsonl`。
- adapter 结果：`/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/adapter-nonthinking-128-20260703.jsonl`。
- base：`256` generated tokens，runtime `53.7887s`，tokens/s `4.7594`。
- adapter：`240` generated tokens，runtime `77.8954s`，tokens/s `3.0811`。
- 判断：`--enable_thinking false` 已传入，但 ms-swift/Qwen 模板仍保留空的 `<think></think>` 前缀；adapter 两条输出完整结束，base 两条仍在 128 token 上限处截断。

已完成半真实 100 step DPO 试跑：

- 新增数据生成脚本：`scripts/make_ops_dpo.py`。
- 新增数据：`datasets/ops_dpo_512.jsonl`，共 `512` 条，主题覆盖 checkpoint、resume、共享服务器安全、框架评估、adapter 导出、实验记录和网络约束。
- 新增启动脚本：`scripts/run_ms_swift_ops_dpo_100step.sh`。
- 输出目录：`/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-ops-100step/ops-100step-20260703/v0-20260703-141605`。
- 训练结果：`global_step/max_steps=100/100`，`train_runtime=344.9065s`，`train_samples_per_second=2.319`，`train_steps_per_second=0.29`，`train_loss=0.01584221`，显存记录 `52.56 GiB`。
- 最终 step：`loss=1.607e-07`，`rewards/accuracies=1.0`，`rewards/margins=18.0`。
- adapter：`checkpoint-100/adapter_model.safetensors` 约 `223M`，`scripts/inspect_adapter.py` 检查通过，包含 `992` 个 LoRA tensor。
- 固定 prompts 对照：base 和 100-step adapter 均 exit `0`，两条 prompts 输出完全一致；base `3.0675 tokens/s`，adapter `3.0334 tokens/s`。
- 判断：100 step 训练、保存、加载链路通过；但该半真实模板数据很容易拟合，固定 prompts 暂无可见输出差异，不能证明真实效果提升。

仍需继续评估：

- tiny 数据只能说明训练链路跑通，不能代表最终效果。
- 合成 256 条数据只能说明短程稳定性和可优化性，不能代表真实 DPO 效果。
- 半真实 `ops_dpo_512` 可以说明 100 step 训练和 adapter 产物路线，但仍不能代表业务泛化效果。
- 当前 FSDP2 sharded checkpoint 可以保存，但恢复链路未通过；正式训练可以先采用 `FULL_STATE_DICT + save_only_model` 的 adapter 导出路线作为最低限度产物保障，该 adapter 导出路线已完成重新加载推理 smoke test。
- 当前 `learning_rate=0.0` 是 1 step + 默认调度下的 smoke 现象，正式训练需要设置 warmup/scheduler。
- 推理日志出现 `chunk_gated_delta_rule` tensor shape warning，base 和 adapter 都会出现；后续长训或正式评测前需要确认该 warning 是否影响正确性/效率。
- 固定 prompts 在 128 token 下仍可能截断 base 输出，且 100-step adapter 对这 2 条 prompts 无可见差异；正式评测需要更真实的验证集、更长 `MAX_NEW_TOKENS`，以及后处理移除空的 `<think></think>` 前缀。
- 官方 NPU 文档里 DPO 已验证组合偏 `deepspeed`；本次我们实际跑通的是 FSDP2，需要继续做更长步数和真实数据评估。
- `decord` 未安装，对纯文本 DPO 不是阻塞；若后续做多模态/视频数据会成为依赖项。

## 服务器网络约束

服务器只能访问部分中国大陆网站。后续服务器侧依赖默认使用大陆镜像、ModelScope、GitCode/Gitee；GitHub/Hugging Face 资料优先在本地获取后通过 `scp` 同步到服务器我们的工作区。

## ms-swift 实测结论

`ms-swift` 是目前对 Qwen3.6-27B 支持证据最强的训练框架。

已通过：

- `transformers==5.12.1` 可识别 `model_type=qwen3_5`
- 可在 meta device 上构建 `Qwen3_5ForConditionalGeneration`
- `ms-swift` 可识别本地 `/models/Qwen3.6-27B` 为 `qwen3_5`
- 关闭 NPU model patch 后，processor/tokenizer/chat_template 加载通过

已修复：

- 安装 `triton-ascend==3.2.1`。
- 从 GitCode 安装 `MindSpeed core_r0.16.0`，替换容器原有 `mindspeed 0.12.1`。
- 默认 NPU model patch 现在可以加载，并将 Qwen3.5/Qwen3.6 的 `chunk_gated_delta_rule` 指向 ms-swift 内置 MindSpeed 实现。

已通过：

- `swift rlhf --help` 正常。
- `ms-swift + Qwen3.6-27B + DPO + LoRA + FSDP2` 跑通 1 step。
- 训练输出目录：`/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-smoke/v1-20260703-094330`。
