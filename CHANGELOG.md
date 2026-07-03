# 更新说明

## v0.1.12 - 2026-07-03

新增：

- `scripts/run_ms_swift_fixed_prompt_eval.sh` 支持：
  - `ENABLE_THINKING`
  - `PRESERVE_THINKING`
  - `TEMPERATURE`
- `scripts/run_ms_swift_base_adapter_compare.sh` 同步透传上述推理参数。

固定 prompts non-thinking 128 token 对照：

- 运行参数：
  - `RUN_ID=nonthinking-128-20260703`
  - `MAX_NEW_TOKENS=128`
  - `ENABLE_THINKING=false`
- base 结果：
  - `result_path=/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/base-nonthinking-128-20260703.jsonl`
  - `num_generated_tokens=256`
  - `runtime=53.7887s`
  - `tokens/s=4.7594`
- adapter 结果：
  - `result_path=/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/adapter-nonthinking-128-20260703.jsonl`
  - `num_generated_tokens=240`
  - `runtime=77.8954s`
  - `tokens/s=3.0811`

当前判断：

- `--enable_thinking false` 已确认传入 ms-swift 命令。
- 输出不再展开思考内容，但仍保留空的 `<think></think>` 前缀；这属于模板行为，不能把 `enable_thinking=false` 理解为完全删除 thinking 标签。
- adapter 两条输出均完整结束，base 两条输出仍在 128 token 上限处截断。
- adapter 推理速度仍低于 base，小样本观测约为 `3.08` vs `4.76 tokens/s`。

## v0.1.11 - 2026-07-03

新增：

- 新增 `datasets/fixed_eval_prompts.jsonl`，包含 2 条固定 prompts，用于 base/adapter 对照。
- 新增 `scripts/run_ms_swift_fixed_prompt_eval.sh`：
  - `ADAPTER_PATH` 为空时跑 base。
  - `ADAPTER_PATH` 非空时加载 LoRA adapter。
  - 支持 `RESULT_PATH` 指定输出文件。
- 新增 `scripts/run_ms_swift_base_adapter_compare.sh`，封装固定 prompts 的 base/adapter 顺序对照，分别保存日志、退出码和结果文件。

固定 prompts 对照实测：

- base 结果：
  - `/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/base-20260703.jsonl`
- adapter 结果：
  - `/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/adapter-20260703.jsonl`
- 两边均完成 2 条 prompts、各生成 `128` tokens。
- base：
  - runtime `27.8733s`
  - tokens/s `4.5922`
  - samples/s `0.0718`
- adapter：
  - runtime `42.1431s`
  - tokens/s `3.0373`
  - samples/s `0.0475`

当前判断：

- base 和 adapter 固定 prompts 输出方向基本一致；2 step 合成 DPO adapter 太小，不能期待明显效果差异。
- adapter 推理比 base 慢，当前小样本观测约为 `3.04` vs `4.59` tokens/s。
- 两边各出现 2 次 `chunk_gated_delta_rule` tensor shape warning，说明该 warning 不是 adapter 独有。
- 64 tokens 仍会截断部分输出；正式评测需要禁用 thinking 或提高 `MAX_NEW_TOKENS`。

## v0.1.10 - 2026-07-03

新增：

- 新增 `datasets/tiny_infer.jsonl`，用于非交互式 adapter 推理 smoke test。
- 新增 `scripts/run_ms_swift_adapter_infer_smoke.sh`，默认加载上一版导出的 LoRA adapter：
  - `/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step/v0-20260703-130918/checkpoint-2`

adapter 重新加载推理测试：

- 命令在 `llin-rl-dpo` 容器内运行，使用 `swift infer`。
- 参数：
  - `--model /models/Qwen3.6-27B`
  - `--model_type qwen3_5`
  - `--adapters <checkpoint-2>`
  - `--infer_backend pt`
  - `--device_map auto`
  - `--max_new_tokens 16`
  - `--val_dataset /workspace/llin-rl-dpo/datasets/tiny_infer.jsonl`
- exit code 为 `0`。
- 结果文件：
  - `/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step/v0-20260703-130918/checkpoint-2/infer_result/20260703-132529.jsonl`

推理指标：

- prompt tokens: `37`
- generated tokens: `16`
- runtime: `49.2921s`
- samples/s: `0.0203`
- tokens/s: `0.3246`

当前判断：

- `FULL_STATE_DICT + save_only_model=true` 导出的 LoRA adapter 已验证可以被 ms-swift 重新加载并生成。
- 16 token smoke 输出被截断，只说明加载/推理链路可用，不代表效果。
- 日志出现 `chunk_gated_delta_rule` tensor shape warning，推理仍完成；后续长训或正式评测前需要单独确认该 warning 是否影响正确性/效率。
- 默认 FSDP2 sharded checkpoint resume 仍未通过；当前可靠产物路线是导出普通 LoRA adapter。

## v0.1.9 - 2026-07-03

新增：

- 新增 `patches/sitecustomize.py`，提供可选 runtime patch：
  - `LLIN_SWIFTMODEL_ASSIGN_PATCH=1`
  - 让 `SwiftModel.load_state_dict` 接受 `assign=` 参数，并在普通加载时传给底层 `base_model.load_state_dict`。
- 新增 `configs/fsdp2_full_state.json`，用于 FSDP2 `FULL_STATE_DICT` 保存对照测试。
- 新增 `scripts/inspect_adapter.py`，用于读取并检查 LoRA adapter 配置和 safetensors 权重。
- `scripts/run_ms_swift_qwen36_dpo_smoke.sh` 增加：
  - `FSDP_CONFIG`
  - `SAVE_ONLY_MODEL`
  - `LLIN_SWIFTMODEL_ASSIGN_PATCH`

resume 排查：

- 启用 `LLIN_SWIFTMODEL_ASSIGN_PATCH=1` 后，从 `checkpoint-10` resume 不再卡在：
  - `SwiftModel.load_state_dict() got an unexpected keyword argument 'assign'`
- 但继续失败于：
  - `Missing key in checkpoint state_dict: model.model.visual.patch_embed.proj.weight`

判断：

- 默认 FSDP2 sharded checkpoint 保存的是 adapter/trainable 状态，不包含完整视觉塔等冻结底座权重。
- FSDP2 resume loader 期望完整模型 sharded state，因此仍无法恢复。
- 这说明 `assign` patch 只能解决签名兼容第一层问题，不足以让默认 FSDP2 sharded checkpoint 可恢复。

adapter 导出测试：

- 使用 `configs/fsdp2_full_state.json`。
- 参数：
  - `MAX_STEPS=2`
  - `SAVE_STRATEGY=steps`
  - `SAVE_STEPS=1`
  - `SAVE_TOTAL_LIMIT=1`
  - `SAVE_ONLY_MODEL=true`
  - `FSDP_CONFIG=/workspace/llin-rl-dpo/configs/fsdp2_full_state.json`
- 输出目录：
  - `/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step/v0-20260703-130918/checkpoint-2`

adapter 导出结果：

- 训练 exit code 为 `0`。
- 生成普通 LoRA adapter 文件：
  - `adapter_config.json`
  - `adapter_model.safetensors`
- `adapter_model.safetensors` 约 `223M`。
- `scripts/inspect_adapter.py` 检查通过：
  - `adapter_type=LORA`
  - `base_model=/models/Qwen3.6-27B`
  - `num_tensors=992`

当前判断：

- 默认 FSDP2 sharded checkpoint resume 仍未通过。
- `FULL_STATE_DICT + save_only_model=true` 可以作为当前最低限度可用产物保存路线，能导出普通 LoRA adapter。
- 下一步应测试该 adapter 是否能重新加载做推理/评测，并继续调查 sharded checkpoint resume 的根修复。

## v0.1.8 - 2026-07-03

新增：

- `scripts/run_ms_swift_qwen36_dpo_smoke.sh` 增加环境变量配置：
  - `NUM_TRAIN_EPOCHS`
  - `SAVE_STRATEGY`
  - `SAVE_STEPS`
  - `SAVE_TOTAL_LIMIT`
  - `EVAL_STRATEGY`
  - `RESUME_FROM_CHECKPOINT`

实测：

- 使用 `ms-swift + Qwen3.6-27B + DPO + LoRA + FSDP2 + 8 NPU` 路线。
- 使用 `datasets/synthetic_dpo_256.jsonl`。
- 跑 20 step checkpoint 保存测试：
  - `MAX_STEPS=20`
  - `SAVE_STRATEGY=steps`
  - `SAVE_STEPS=10`
  - `SAVE_TOTAL_LIMIT=2`
- 输出目录：`/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-checkpoint-20step/v0-20260703-124743`

保存测试结果：

- 训练 exit code 为 `0`。
- `global_step/max_steps=20/20`
- `train_runtime=97.8253s`
- `train_samples_per_second=1.636`
- `train_steps_per_second=0.204`
- `memory(GiB)=51.85`
- 生成 `checkpoint-10` 和 `checkpoint-20`。
- 每个 checkpoint 约 `713M`。
- checkpoint 形态为 FSDP2 sharded checkpoint：
  - `pytorch_model_fsdp_0`
  - `optimizer_0`
  - `scheduler.pt`
  - `trainer_state.json`
  - `rng_state_0.pth` 到 `rng_state_7.pth`

恢复测试结果：

- 尝试从 `checkpoint-10` 恢复并跑到 `max_steps=12`。
- 训练 exit code 为 `1`。
- 失败点：
  - `TypeError: SwiftModel.load_state_dict() got an unexpected keyword argument 'assign'`
- 发生位置：
  - `accelerate.utils.fsdp_utils.fsdp2_load_full_state_dict`
  - `model.load_state_dict(sharded_sd, assign=True)`

当前判断：

- FSDP2 checkpoint 保存链路通过。
- FSDP2 checkpoint resume 链路未通过。
- 正式长训前必须解决恢复问题，或采用可验证的 adapter 导出/保存方案作为备选。

## v0.1.7 - 2026-07-03

新增：

- 新增 `scripts/make_synthetic_dpo.py`，用于生成可复现的合成 DPO JSONL 数据。
- 新增 `datasets/synthetic_dpo_256.jsonl`，包含 256 条合成偏好样本，用于短程稳定性和吞吐 smoke。

实测：

- 使用上一版跑通的 `ms-swift + Qwen3.6-27B + DPO + LoRA + FSDP2 + 8 NPU` 路线。
- 数据集从 16 条 tiny DPO 扩展到 256 条合成 DPO。
- 设置 `MAX_STEPS=20`，输出目录为 `/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-stability-20step/v0-20260703-120434`。
- 训练任务 exit code 为 `0`。

结果：

- `global_step/max_steps=20/20`
- `train_runtime=118.3962s`
- `train_samples_per_second=1.351`
- `train_steps_per_second=0.169`
- 平均 step time 约 `5.918s/it`
- `memory(GiB)=51.93`
- `train_loss=0.0722111`
- 最后一步 `loss=5.117e-05`
- 最后一步 `rewards/accuracies=1.0`
- 最后一步 `rewards/margins=10.25`

判断：

- FSDP2 路线不只是单步可跑，至少在 20 step 合成数据上完成了短程稳定性测试。
- 1 step smoke 的 `139.6s/step` 主要受启动、首次 batch、编译/初始化影响；20 step 运行中平均约 `5.92s/step` 更接近短程训练速度。
- 该数据集是合成数据，loss 和 reward margin 不能代表真实偏好效果；下一步必须接真实或半真实 DPO 数据和验证集。

## v0.1.6 - 2026-07-03

新增：

- 新增 `datasets/tiny_dpo.jsonl`，提供 16 条标准 `messages` + `rejected_response` 的 tiny DPO smoke 数据。
- 新增 `scripts/run_ms_swift_qwen36_dpo_smoke.sh`，固化 `ms-swift + Qwen3.6-27B + DPO + LoRA + FSDP2` 的 8 NPU 最小训练命令。
- 新增 `scripts/env_report.py`，用于记录容器内关键包版本和 NPU 可见数量。

环境推进：

- 记录官方 `quay.io/ascend/ms-swift:v4.3.0-A3-py311-CANN9.0.0-ubuntu22.04` 镜像 tag 存在且 manifest 支持 arm64，但服务器下载大层不稳定，未作为本轮主路线。
- 在现有 `llin-rl-dpo` 容器中安装 `triton-ascend==3.2.1`。
- 从 GitCode 拉取并安装 `MindSpeed core_r0.16.0`，替换容器原有 `mindspeed 0.12.1`。
- `ms-swift` 默认 NPU model patch 已能正常加载 Qwen3.5/Qwen3.6 linear attention 的 MindSpeed 实现。

实测：

- `swift rlhf --help` 通过。
- 首次 `--deepspeed zero2` smoke 因容器内未安装 `deepspeed` 失败，训练未进入模型加载阶段。
- 切换到 `--fsdp fsdp2` 后，4 条数据版本成功加载模型并初始化 LoRA/FSDP2，但 8 rank 分片后没有形成有效训练 step，停在 `global_step=0`。
- 扩展 tiny 数据到 16 条后，`ms-swift + Qwen3.6-27B + DPO + LoRA + FSDP2` 完成 1 个 optimizer step。

结果：

- 输出目录：`/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-smoke/v1-20260703-094330`
- `global_step/max_steps=1/1`
- `loss=0.69140625`
- `grad_norm=1.2919271`
- `train_runtime=139.6129s`
- `train_samples_per_second=0.057`
- `train_steps_per_second=0.007`
- `memory(GiB)=51.19`
- 参数规模：`27415.0925M`，其中 LoRA 可训练参数 `58.3639M`，约 `0.2129%`

当前判断：

- 当前服务器和我们自己的 `llin-rl-dpo` 容器已经能跑通 Qwen3.6-27B 的 DPO 最小训练链路。
- 这只是 smoke test，不代表最终效率或效果；下一步应换真实 DPO 数据，跑 20-100 step 级别的稳定性/吞吐评估，并补充验证集偏好指标。
- 官方 NPU 文档里 DPO 已验证组合偏 `deepspeed`，但本轮实际跑通的是 FSDP2；后续需评估 FSDP2 长步数稳定性，并决定是否安装/测试 deepspeed。

## v0.1.5 - 2026-07-03

新增：

- 新增 `scripts/ms_swift_qwen36_probe.py`，用于在不加载 27B 权重的情况下探测 ms-swift/Transformers 对本地 Qwen3.6-27B 的支持。

实测：

- 记录服务器只能访问部分中国大陆网站，后续服务器侧依赖默认使用大陆镜像、ModelScope、GitCode/Gitee，本地资料通过 `scp` 同步。
- 将 `ms-swift` 源码从本地打包同步到服务器我们的工作区。
- 在 `llin-rl-dpo` 容器内补齐 ms-swift framework 依赖，升级 `transformers==5.12.1` 和 `mistral-common==1.11.5`。
- 卸载 CUDA 版 `torchaudio==2.11.0`，解决 Transformers 5.12 import 链路寻找 `libcudart.so.13` 的问题。

结果：

- Transformers 5.12 能识别 `qwen3_5`，并能在 meta device 上构建 `Qwen3_5ForConditionalGeneration`。
- ms-swift 能识别本地 `/models/Qwen3.6-27B` 为 `qwen3_5`，模板为 `qwen3_5`。
- 关闭 ms-swift NPU model patch 后，Qwen3.6 processor 加载通过，tokenizer 和 chat_template 正常。

当前阻塞：

- 默认开启 ms-swift NPU model patch 时，Qwen3.5/Qwen3.6 linear attention 的 MindSpeed Triton 路径在当前容器中编译失败。
- 当前容器是 `torch_npu 2.7.1.post4`、`mindspeed 0.12.1`、`triton 3.2.0`、CANN 9.0.0；与 ms-swift Qwen3.5 NPU patch 文档里的验证组合不一致。
- 还不能开始正式 DPO 训练；下一步应寻找或构建 ms-swift Qwen3.6 适配版本组合的 Ascend 容器。

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
