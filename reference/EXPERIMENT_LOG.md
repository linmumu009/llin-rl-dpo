# qwen3.6-27b DPO 训练框架实验日志

日期：2026-07-03

## 目标

在当前 Ascend 服务器上，对 qwen3.6-27b 的 DPO 训练方案做框架可行性测试，比较：

- 能否跑通：环境、数据预处理、权重格式、单步训练、保存/恢复。
- 效率：samples/s、step time、HBM、AICore 利用率、通信/重切分开销。
- 效果：DPO loss、chosen/rejected reward margin、验证集偏好准确率、必要时做人工/自动评测。

## 约束

- 不改动物理机系统配置。
- 不修改、不停止、不复用别人已有 Docker 容器。
- 我们只新建并使用容器名：`llin-rl-dpo`。
- 我们自己的服务器工作区固定为：`/data/liulin/llin-rl-dpo`。
- 本地外部框架资料放在：`reference/`。

## 已完成：服务器只读基线检查

连接目标：

- 主机：`182.151.17.158`
- SSH 端口：`22205`
- 用户：`root`
- 登录方式：本地私钥，见根目录 `SSH_PRIVATE_KEY_LOGIN.md`

宿主机：

- hostname：`XXZX2JL-501-F-04-A1P1-SEV-HWA800A3-10U05`
- OS：`ctyunos 23.01 3`
- Docker：`28.5.2`
- Docker runtime：存在 `ascend` runtime，路径为 `/usr/local/Ascend/Ascend-Docker-Runtime/ascend-docker-runtime`
- Python：宿主机 `3.13.13`，训练环境以容器内 Python 为准

NPU：

- `npu-smi info` 可用，版本 `26.0.rc1`
- 设备：8 个 Ascend910 NPU，每个逻辑 NPU 显示 2 个 chip，共 `/dev/davinci0` 到 `/dev/davinci15`
- HBM：每 chip `65536 MB`
- 当前训练进程：未发现 NPU running processes

磁盘：

- `/`：300G，总用量约 42G，剩余约 259G
- `/data`：3.5T，总用量约 2.3T，剩余约 1.3T

已有容器边界：

- 存在 `llin-autoresearch-msllm`、`llin-autoresearch`、`mindspeed-llm-26.0.0-a3-sshd-yehairui`、`mindspeed_mm_rjx`、K8s 系统容器等。
- 没有发现名为 `llin-rl-dpo` 的容器。
- 已有容器只作为只读参考，不修改。

可用镜像：

- `swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed_rl_pt25_25rc3:2.2.0-A3-ARM`
- `mindspeed-llm:26.0.0-a3-sshd`
- `swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64`
- 多个 `vllm-ascend` 和 `qwen3.6-27b:vllm-ascend-v0.18.0*` 推理镜像

模型：

- HF 权重目录：`/data/models/Qwen3.6-27B`
- 目录包含 `config.json`、`tokenizer.json`、`model-00001-of-00015.safetensors` 等。
- `config.json` 显示：
  - `architectures`: `Qwen3_5ForConditionalGeneration`
  - `model_type`: `qwen3_5`
  - `language_model_only`: `false`
  - 包含 `text_config` 与 `vision_config`
  - 文本侧：64 层、hidden size 5120、attention heads 24、KV heads 4、vocab size 248320

重要判断：

- 当前目标模型不是 MindSpeed-RL 现成样例里的普通 `qwen3_30b_a3b`，而是 `qwen3_5` conditional generation / 多模态配置。
- MindSpeed-RL 的 DPO 样例不能直接照抄跑 qwen3.6-27B，需要先验证模型架构支持、权重转换支持和 tokenizer/template 兼容性。

## 已完成：本地参考资料

已拉取官方仓库：

- `reference/MindSpeed-RL`
- 来源：https://github.com/Ascend/MindSpeed-RL

官方 README 当前关键信息：

- MindSpeed-RL 是昇腾生态 RL 训练框架。
- CLI 支持 `DAPO/DPO/GRPO/PPO`。
- README 标注 DPO 支持模型为 `Qwen3-30B-A3B`，状态为 `Preview`。
- README 提到 2026.4 后 MindSpeed-RL 暂停新增功能集成，建议关注 verl 昇腾实践。

本地参考文件：

- `reference/MindSpeed-RL/docs/zh/algorithms/dpo.md`
- `reference/MindSpeed-RL/configs/dpo_qwen3_30b_a3b_A3.yaml`
- `reference/MindSpeed-RL/examples/dpo/dpo_trainer_qwen3_30b_a3b.sh`

## MindSpeed-RL DPO 官方样例摘要

数据：

- 官方以 `orca_dpo_pairs` 为例。
- 需要先用 `examples/data/preprocess_data.sh` 做数据预处理。
- 预处理配置示例：`configs/datasets/orca_rlhf.yaml`

权重：

- DPO 使用 Megatron-mcore 格式权重。
- HF safetensors 需要先转换为 mcore。
- Actor 与 Reference 可以使用同一份初始权重。

训练：

- 样例脚本默认 `GPUS_PER_NODE=16`、`NNODES=2`。
- 样例配置默认 `tensor_model_parallel_size=2`、`pipeline_model_parallel_size=8`。
- 样例模型为 `qwen3_30b_a3b`，不是当前 `qwen3_5`。

官方性能参考：

- Qwen3-30B-A3B，Atlas 900 A3 SuperPoD，2x8，GBS 64，dynamic sequence：`7.19 samples/s`。
- 该数据仅作为参考，不能直接代表 qwen3.6-27B。

## 下一步计划

1. 创建服务器工作区 `/data/liulin/llin-rl-dpo`。
2. 用 `mindspeed_rl_pt25_25rc3:2.2.0-A3-ARM` 新建 `llin-rl-dpo` 容器。
3. 容器内做环境 smoke test：
   - `python --version`
   - `python -c "import torch; import torch_npu; print(torch.__version__)"`
   - `npu-smi info`
   - `python -c "import mindspeed_rl"`
4. 在容器内同步/复制 MindSpeed-RL 参考代码到我们的工作区。
5. 验证 qwen3.6-27B 的 `qwen3_5` 架构是否被当前 transformers / MindSpeed / MindSpeed-LLM 支持。
6. 若 MindSpeed-RL 不能直接支持 qwen3.6-27B：
   - 记录失败原因。
   - 转向 verl NPU/FSDP 或 torch_npu FSDP smoke test。
7. 最小训练评估路径：
   - 先用小样本 DPO 数据集跑 preprocessing。
   - 再做权重转换 dry-run 或小模型替代验证。
   - 最后尝试 qwen3.6-27B 单步/少步 DPO。

