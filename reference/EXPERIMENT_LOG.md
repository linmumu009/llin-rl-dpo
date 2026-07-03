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

## 2026-07-03 容器创建与 smoke test

已创建服务器工作区：

- `/data/liulin/llin-rl-dpo/logs`
- `/data/liulin/llin-rl-dpo/configs`
- `/data/liulin/llin-rl-dpo/scripts`
- `/data/liulin/llin-rl-dpo/outputs`
- `/data/liulin/llin-rl-dpo/datasets`
- `/data/liulin/llin-rl-dpo/reference`

### 尝试 1：MindSpeed-RL 专用镜像

镜像：

- `swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed_rl_pt25_25rc3:2.2.0-A3-ARM`

发现：

- 默认 `python` 是 `/usr/local/bin/python`，只有 `pip` 和 `setuptools`，没有 torch。
- 实际训练环境在 `/root/miniconda3/envs/mindspeed_rl_2.2.0`。
- 源码在 `/env/MindSpeed-RL`。
- `torch` 版本：`2.5.1`
- `torch_npu` 版本：`2.5.1.post4.dev20250922`

问题：

- 未 source Ascend toolkit 时，import torch_npu 缺 `libhccl.so`。
- source `/usr/local/Ascend/ascend-toolkit/set_env.sh` 后，缺 `libascend_hal.so`。
- 手动补充 `LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH` 后可以 import torch_npu。
- 但 `torch_npu.npu.device_count()` 返回 `0`。
- 单设备 `/dev/davinci0` 和全设备 `/dev/davinci0-15` 暴露都未解决。

判断：

- 该镜像能作为代码/环境参考，但当前与宿主 Ascend runtime/driver 的设备可见性没有打通。

### 尝试 2：MindSpeed-LLM 26.0.0 A3 镜像

镜像：

- `swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64`

当前保留的容器：

- 名称：`llin-rl-dpo`
- 状态：running
- privileged：false
- 工作区：`/workspace/llin-rl-dpo`
- 模型只读挂载：`/models/Qwen3.6-27B`

发现：

- Python：`3.11.14`
- Python 路径：`/opt/conda/bin/python`
- torch：`2.7.1+cpu`
- torch_npu：`2.7.1.post4`
- 镜像内环境变量已经包含 CANN 9.0.0、driver lib64/common/driver、ATB/NNAL 等路径。
- 可以 import torch 与 torch_npu。

问题：

- `torch_npu.npu.device_count()` 仍返回 `0`。
- 单设备和全设备挂载都未解决。
- `npu-smi` 在容器内不在 PATH；宿主机路径是 `/usr/local/sbin/npu-smi`。

### 未执行项

尝试用 `--privileged` 重建容器做隔离实验时，被安全审查拦截。

原因：

- `--privileged` 会显著扩大容器对宿主机的访问权限。
- 在没有用户明确确认该风险前，不执行该方案。

### 当前结论

`llin-rl-dpo` 容器已经按约束创建并运行，但训练前置 smoke test 未完全通过。

当前阻塞点：

- 容器内 torch_npu 能 import，但 NPU device_count 为 `0`。

下一步建议：

1. 确认 Ascend Docker runtime 是否需要额外设备、annotation、用户组或 Kubernetes device plugin 分配方式。
2. 如用户明确接受风险，再做一次 `--privileged` 对照实验，判断是否为容器权限边界导致。
3. 若 privileged 也失败，则转向宿主 Ascend runtime/driver 配置排查。

## 2026-07-03 NPU 可见性修复

### 根因

Ascend Docker Runtime 的 README 明确说明：

- prestart hook 会根据 `ASCEND_VISIBLE_DEVICES` 挂载 NPU 设备。
- hook 同时配置 device cgroup。
- hook 还会挂载 Host 侧 CANN Runtime Library。

此前只设置 `ASCEND_RT_VISIBLE_DEVICES`，或手动传入 `--device=/dev/davinci*`，没有正确触发 Ascend Docker Runtime 的完整设备注入链路。

### 关键现象

1. 使用 `ASCEND_VISIBLE_DEVICES=0` 重建容器后：
   - `docker inspect` 中 `HostConfig.Devices=[]`，说明设备由 runtime hook 注入，不再表现为手工 `--device`。
   - 容器内出现 `/usr/local/bin/npu-smi`。
   - `npu-smi info` 报 `device is used`。

2. 只读检查已有容器后发现：
   - `llin-autoresearch-msllm` 已绑定 `/dev/davinci0`。
   - `llin-autoresearch` 已绑定 `/dev/davinci0`。
   - 因此设备 `0` 不能作为我们的首选测试设备。

3. 使用 `ASCEND_VISIBLE_DEVICES=1` 重建 `llin-rl-dpo` 后：
   - `torch_npu.npu.device_count()` 返回 `1`。
   - 最小 NPU tensor 计算成功：
     - 输入：`torch.ones(4).npu().sum().cpu().item()`
     - 输出：`4.0`

### 当前保留容器

容器：

- 名称：`llin-rl-dpo`
- 镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64`
- `ASCEND_VISIBLE_DEVICES=1`
- `ASCEND_RT_VISIBLE_DEVICES=0`
- 工作区：`/workspace/llin-rl-dpo`
- 模型只读挂载：`/models/Qwen3.6-27B`

验证：

- Python：`3.11.14`
- torch：`2.7.1+cpu`
- torch_npu：`2.7.1.post4`
- NPU device count：`1`
- 最小 NPU 计算：通过

### 新结论

NPU 容器底座已经可用，当前是单 NPU smoke test 通过。

训练 qwen3.6-27B DPO 前仍需处理：

1. 多 NPU 设备集合确认，避免占用已有容器分配的设备。
2. qwen3.6-27B 的 `qwen3_5` 架构是否被 MindSpeed-LLM/MindSpeed-RL 支持。
3. HF safetensors 到训练框架所需权重格式的转换方案。
4. DPO 数据格式与预处理配置。

## 2026-07-03 8 NPU smoke test

### 设备选择

现有容器占用情况：

- `llin-autoresearch-msllm` 绑定 `/dev/davinci0`
- `llin-autoresearch` 绑定 `/dev/davinci0`
- `ASCEND_VISIBLE_DEVICES=0` 会触发 Ascend Runtime hook，但容器内 `npu-smi` 报 `device is used`

为避免影响已有容器，`llin-rl-dpo` 改用：

- `ASCEND_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`

当前容器：

- 名称：`llin-rl-dpo`
- 镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-llm:26.0.0-a3-openeuler24.03-py3.11-aarch64`
- 工作区：`/workspace/llin-rl-dpo`
- 模型只读挂载：`/models/Qwen3.6-27B`

### 单进程多卡可见性

结果：

- torch：`2.7.1+cpu`
- torch_npu：`2.7.1.post4`
- device_count：`8`

### 逐卡 tensor smoke test

脚本：

- `scripts/npu_multicard_smoke.py`

结果：

```text
device_count=8
device=0 sum=4.0
device=1 sum=4.0
device=2 sum=4.0
device=3 sum=4.0
device=4 sum=4.0
device=5 sum=4.0
device=6 sum=4.0
device=7 sum=4.0
```

### HCCL all-reduce smoke test

脚本：

- `scripts/hccl_smoke.py`

命令：

```bash
torchrun --nproc_per_node 8 --master_port 29591 scripts/hccl_smoke.py
```

结果：

```text
world_size=8 all_reduce_sum=36.0 expected=36.0
```

结论：

- 8 逻辑 NPU 可见。
- 每张逻辑 NPU 可执行 tensor 计算。
- 8 进程 HCCL all-reduce 通信正常。

## 2026-07-03 qwen3.6-27B 训练前置检查

### Transformers

容器内版本：

- Transformers：`4.57.1`

检查命令：

```bash
TRANSFORMERS_OFFLINE=1 python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('/models/Qwen3.6-27B', trust_remote_code=True)"
```

结果：

- 失败。
- 报错：`model type qwen3_5` 不被 Transformers 识别。

### 本地模型元数据

`/data/models/Qwen3.6-27B/config.json`：

- `architectures`: `Qwen3_5ForConditionalGeneration`
- `model_type`: `qwen3_5`
- `language_model_only`: `false`
- 包含 `text_config` 和 `vision_config`

`/data/models/Qwen3.6-27B/configuration.json`：

```json
{"framework":"Pytorch","task":"image-text-to-text"}
```

### MindSpeed-LLM

FSDP2 入口：

- `train_fsdp2.py` 支持 `--stage dpo`
- `--model-type-hf` 可选项包含 `qwen3`、`qwen3-moe`、`qwen3-next`
- 当前没有直接的 `qwen3_5` 选项

关键观察：

- FSDP2 help 中有 `full_attention_interval`、`linear_key_head_dim`、`partial_rotary_factor` 等与 qwen3.6 text_config 接近的参数。
- MindSpeed-LLM 内部有 `qwen3-next` FSDP2 模型实现，但 qwen3.6 的 config 是 `qwen3_5`，且是 image-text-to-text conditional generation。

### vLLM Ascend 参考

服务器已有：

- `/data/models/qwen3.6_27b_docker_compose`
- 镜像：`qwen3.6-27b:vllm-ascend-v0.18.0-a3`

compose 文件引用的官方文档：

- `https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html`

官方文档信息：

- Qwen3.6-27B 是 dense model，使用 hybrid attention design，包含 GDN + full attention。
- Qwen3.6-27B 在 vLLM Ascend 中 first supported in `v0.18.0rc1`。
- BF16 版本要求 1 Atlas 800 A3 `(64G x 16)` 或 1 Atlas 800 A2 `(64G x 8)` 节点。

结论：

- 推理路径已有可用参考。
- 训练路径不能直接复用 vLLM。
- 下一步需要：
  1. 找到或移植 qwen3_5 的 Transformers/MindSpeed 训练模型类。
  2. 或把 qwen3.6 text_config 安全映射到 MindSpeed `qwen3-next` 路线做 dry-run。
  3. 或先用 MindSpeed 已支持的 Qwen3/Qwen3-Next 模型做框架效率基准，再回到 qwen3.6 适配。

## 2026-07-03 外部框架扩展调查

用户要求不要只考虑 MindSpeed 和 FSDP，因此补充调查了 Ascend + Qwen3.6 + DPO 相关训练框架。

详细结果：

- `reference/ASCEND_QWEN36_FRAMEWORK_SURVEY.md`

关键结论：

1. ModelScope `ms-swift` 是下一轮第一优先级。
   - 官方资料同时覆盖 Qwen3.6、DPO、人类对齐、Ascend NPU、FSDP/FSDP2、DeepSpeed、Megatron。
   - NPU 最佳实践里明确列出 DPO 已支持，并给出 Ascend NPU 环境准备、训练、推理、部署路径。
   - 还有 Qwen3.5 最佳实践和 FLA patch 说明，和我们本地 `qwen3_5` 配置关系更近。

2. LLaMA-Factory NPU 是第二优先级。
   - 官方资料同时覆盖 NPU、Qwen3.6、DPO。
   - 有官方 NPU 镜像和 NPU 训练文档。
   - 仍需实测 `qwen3_5`/Qwen3.6-27B 在 Ascend 上是否能加载和训练。

3. MindSpeed-RL / MindSpeed-LLM 仍保留，但不作为唯一主线。
   - Ascend 原生能力强，DPO 入口存在。
   - 当前直接卡在 `qwen3_5` 模型类支持。

4. verl Ascend 适合后续 RL/GRPO 方向。
   - 官方 Ascend 支持表中已有多种 Qwen RL 组合，包括 Qwen3.5-27B GRPO。
   - 但未看到 Qwen3.6-27B DPO 直接证据。

5. FlagScale / FlagOS / VeRL-FL 作为中长期候选。
   - 目标是统一多芯片训练/RL/推理。
   - 目前未看到 Qwen3.6 + DPO 直接证据。

6. vLLM Ascend 只作为推理和训练前后评测服务。
   - 官方支持 Qwen3.6-27B 推理。
   - 不作为 DPO 训练框架。

下一步执行顺序：

1. 拉取 `ms-swift` 到本地 `reference/ms-swift` 作为不提交的源码参考。
2. 在 `llin-rl-dpo` 容器中做 `ms-swift` import/help/config-load smoke test。
3. 若 `ms-swift` 能加载本地 `/models/Qwen3.6-27B`，准备最小 DPO 数据并跑 1 个 optimizer step。
4. 若 `ms-swift` 卡住，切到 LLaMA-Factory NPU 做同样 smoke test。

## 2026-07-03 ms-swift 初步实测

### 网络约束

用户补充说明：服务器只能访问部分中国大陆网站。

后续服务器侧依赖来源默认按以下优先级处理：

- Python 包：华为云 PyPI、阿里云 PyPI 等大陆镜像。
- 模型与数据：ModelScope 优先，避免直接依赖 Hugging Face。
- Ascend/MindSpeed 源码：GitCode/Gitee 优先。
- GitHub/Hugging Face 资料：优先在本地拉取或查询，再通过 `scp` 同步到服务器我们的工作区。

本轮 `ms-swift` 源码在本地拉取后，打包成 tar 上传到服务器：

- 宿主机目录：`/data/liulin/llin-rl-dpo/reference/ms-swift`
- 容器内目录：`/workspace/llin-rl-dpo/reference/ms-swift`

### 已安装/调整的容器依赖

均只在我们自己的 `llin-rl-dpo` 容器内执行。

新增或升级：

- `modelscope==1.38.0`
- `qwen-vl-utils==0.0.14`
- `ms-swift` 的 `requirements/framework.txt` 依赖
- `transformers==5.12.1`
- `mistral-common==1.11.5`

卸载：

- `torchaudio==2.11.0`

原因：

- `torchaudio 2.11.0` 会在 Transformers 5.12 import 链路中寻找 `libcudart.so.13`，但当前是 Ascend/NPU 容器，不是 CUDA 环境。
- 对当前文本 DPO 路线，`torchaudio` 不是必要依赖。

未解决依赖：

- `decord`：阿里云 PyPI 与华为云 PyPI 当前都没有可安装的 aarch64 包。
- 目前 `decord` 在 processor 加载阶段只是 warning；如果后续处理视频数据才会成为硬阻塞。文本 DPO 暂不应依赖它。

### ms-swift 识别 qwen3.6-27B

使用 `ms-swift` 源码和本地模型路径：

- 模型：`/models/Qwen3.6-27B`
- model type：`qwen3_5`

结果：

- `ms-swift` 能识别本地模型：
  - `swift_info_model_type=qwen3_5`
  - `swift_meta_model_type=qwen3_5`
  - `swift_template=qwen3_5`
- `ms-swift` 对该模型声明的依赖：
  - `transformers>=5.0.0.dev`
  - `qwen_vl_utils>=0.0.14`
  - `decord`

### Transformers qwen3_5 支持

升级到 `transformers==5.12.1` 后：

```text
config_class=Qwen3_5Config
model_type=qwen3_5
architectures=['Qwen3_5ForConditionalGeneration']
meta_model_class=Qwen3_5ForConditionalGeneration
meta_num_parameters=27356728560
```

说明：

- Transformers 5.12 可以识别 qwen3.6-27B 的 `qwen3_5` config。
- 可以在 `accelerate.init_empty_weights()` 下构建 `Qwen3_5ForConditionalGeneration` meta 模型。
- 该测试不加载权重，不占用 NPU 显存。

### Processor 加载

关闭 ms-swift NPU model patch 后，processor 加载通过：

```text
processor_model=None
processor_class=Qwen3VLProcessor
tokenizer_class=Qwen2Tokenizer
has_chat_template=True
```

命令形态：

```bash
PYTHONPATH=/workspace/llin-rl-dpo/reference/ms-swift \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python scripts/ms_swift_qwen36_probe.py \
  --disable-npu-model-patch
```

### 当前 ms-swift 阻塞点

默认开启 ms-swift NPU model patch 时，会触发 Qwen3.5/Qwen3.6 linear attention 的 MindSpeed Triton 路径。

当前容器版本：

- `torch==2.7.1+cpu`
- `torch_npu==2.7.1.post4`
- `triton==3.2.0`
- `mindspeed==0.12.1`
- `transformers==5.12.1`
- CANN：`/usr/local/Ascend/cann-9.0.0`

失败现象：

- `mindspeed.ops.triton.utils` 编译临时 `npu_utils.cpp` 失败。
- 报错包含：`RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` 不是 `rtLimitType_t` 成员。
- 随后又触发 `NameError: name '_cpu_device_warning' is not defined`。

判断：

- 这是 `ms-swift` 的 NPU Qwen3.5/Qwen3.6 patch 与当前 MindSpeed/Triton/CANN 组合不兼容。
- `ms-swift` 文档中的 Qwen3.5 NPU patch 精度对齐组合是 `torch 2.9.0 + MindSpeed 0.16.0 + flash-linear-attention 0.4.2 + triton-ascend 3.2.1 + transformers 5.2.0`，与当前镜像不一致。

当前实事求是结论：

- `ms-swift` 是目前对 Qwen3.6-27B 支持证据最强的训练框架。
- 在当前 MindSpeed-LLM 26.0.0 容器里，ms-swift 的模型识别、Transformers config、meta 模型构建、processor 加载已经通过。
- 还不能开始正式 DPO 训练，因为默认 NPU patch 所需 MindSpeed/Triton/CANN 组合不匹配。
- 下一步应优先寻找或构建一个符合 ms-swift NPU Qwen3.5/Qwen3.6 patch 版本组合的容器；若继续在当前容器中试跑，只能作为关闭 patch 的慢速/风险对照。

## 2026-07-03 ms-swift DPO/FSDP2 smoke 跑通

### 继续解决 NPU patch

官方 `ms-swift` A3 镜像：

- 发现 tag：`quay.io/ascend/ms-swift:v4.3.0-A3-py311-CANN9.0.0-ubuntu22.04`
- manifest 支持 arm64。
- 服务器 `docker pull` 多次卡在大 layer 下载，考虑到服务器只能访问部分中国大陆网站，本轮没有继续依赖该镜像。

在我们自己的 `llin-rl-dpo` 容器中继续修补环境：

- 安装 `triton-ascend==3.2.1`。
- 从 GitCode 拉取 `Ascend/MindSpeed` 的 `core_r0.16.0` 分支到 `/data/liulin/llin-rl-dpo/reference/MindSpeed-core_r0.16.0`。
- `pip install -e /workspace/llin-rl-dpo/reference/MindSpeed-core_r0.16.0` 替换容器原有 `mindspeed 0.12.1`。

修补后：

- `swift rlhf --help` 通过。
- ms-swift 日志显示已 patch Qwen3.5/Qwen3.6 的 `chunk_gated_delta_rule` 到内置 MindSpeed 实现。

### 关键环境版本

容器内版本：

```text
ms-swift=4.5.0.dev0
modelscope=1.38.0
accelerate=1.13.0
peft=0.19.1
trl=0.29.1
mindspeed=0.16.0
triton-ascend=3.2.1
triton=3.5.0
torch=2.7.1+cpu
torch_npu=2.7.1.post4
transformers=5.12.1
numpy=1.26.0
npu_count=8
```

### 数据和脚本

新增本仓库文件：

- `datasets/tiny_dpo.jsonl`
- `scripts/run_ms_swift_qwen36_dpo_smoke.sh`
- `scripts/env_report.py`

tiny DPO 数据使用 ms-swift 标准格式：

- `messages`：system/user/assistant，其中 assistant 是 chosen response。
- `rejected_response`：DPO 的 rejected response。

最小训练命令由脚本固化：

```bash
cd /workspace/llin-rl-dpo
scripts/run_ms_swift_qwen36_dpo_smoke.sh
```

核心参数：

- `--rlhf_type dpo`
- `--model /models/Qwen3.6-27B`
- `--model_type qwen3_5`
- `--tuner_type lora`
- `--target_modules all-linear`
- `--lora_rank 8`
- `--lora_alpha 32`
- `--lora_dropout 0`
- `--max_length 512`
- `--per_device_train_batch_size 1`
- `--max_steps 1`
- `--save_strategy no`
- `--eval_strategy no`
- `--check_model false`
- `--fsdp fsdp2`
- `NPROC_PER_NODE=8`

### 尝试 1：deepspeed zero2

命令使用：

```bash
--deepspeed zero2
```

结果：

- 失败。
- 原因：容器内没有安装 `deepspeed`。
- 报错：`PackageNotFoundError: No package metadata was found for deepspeed`
- 训练未进入模型加载阶段。

判断：

- 这不是 Qwen3.6 模型本身的问题。
- 官方 NPU 文档里 DPO 已验证组合偏 `deepspeed`，后续可单独评估是否安装并测试 deepspeed。

### 尝试 2：FSDP2 + 4 条 tiny 数据

命令切换为：

```bash
--fsdp fsdp2
```

结果：

- 模型权重加载通过。
- Qwen3.6/Qwen3_5 NPU patch 通过。
- LoRA 注入通过。
- FSDP2 初始化通过。
- 但 4 条数据在 8 rank 分片后没有形成有效训练 step，最终 `global_step=0`。

判断：

- 4 条数据对于 8 NPU 分布式 smoke 太少。
- 需要至少保证每个 rank 能分到样本。

### 尝试 3：FSDP2 + 16 条 tiny 数据

将 `datasets/tiny_dpo.jsonl` 扩展到 16 条后重跑。

结果：

```text
global_step/max_steps=1/1
loss=0.69140625
grad_norm=1.2919271
train_runtime=139.6129
train_samples_per_second=0.057
train_steps_per_second=0.007
memory(GiB)=51.19
```

输出目录：

```text
/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-smoke/v1-20260703-094330
```

模型参数信息：

```text
PeftModelForCausalLM: 27415.0925M Params (58.3639M Trainable [0.2129%]), 0.0001M Buffers.
```

结论：

- 在不修改物理机、不动其他用户 Docker 的前提下，我们自己的 `llin-rl-dpo` 容器已经跑通 Qwen3.6-27B 的 DPO 最小训练链路。
- 本次跑通路线是 `ms-swift + Transformers 5.12 + MindSpeed 0.16 + triton-ascend 3.2.1 + LoRA + FSDP2`。
- 这是 smoke test，只说明链路可执行；效率和效果还不能定论。

下一步：

1. 准备真实或半真实 DPO 数据集，先跑 20-100 step 稳定性测试。
2. 记录 step time、HBM、AICore 利用率、通信开销。
3. 增加验证集，记录 DPO loss、chosen/rejected reward margin、偏好准确率。
4. 与 LLaMA-Factory NPU 做同样最小链路对比。
5. 视 FSDP2 长步数结果决定是否安装并测试 deepspeed zero2/zero3。
