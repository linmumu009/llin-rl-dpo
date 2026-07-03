# 更新说明

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
