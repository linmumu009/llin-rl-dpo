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

仍需继续评估：

- tiny 数据只能说明训练链路跑通，不能代表最终效果。
- 合成 256 条数据只能说明短程稳定性和可优化性，不能代表真实 DPO 效果。
- 当前 FSDP2 sharded checkpoint 可以保存，但恢复链路未通过；正式训练可以先采用 `FULL_STATE_DICT + save_only_model` 的 adapter 导出路线作为最低限度产物保障，该 adapter 导出路线已完成重新加载推理 smoke test。
- 当前 `learning_rate=0.0` 是 1 step + 默认调度下的 smoke 现象，正式训练需要设置 warmup/scheduler。
- 推理日志出现 `chunk_gated_delta_rule` tensor shape warning，base 和 adapter 都会出现；后续长训或正式评测前需要确认该 warning 是否影响正确性/效率。
- 固定 prompts 的 64 token 输出仍有截断，正式评测需要禁用 thinking 或提高 `MAX_NEW_TOKENS`。
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
