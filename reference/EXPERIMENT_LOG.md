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

## 2026-07-03 ms-swift DPO/FSDP2 20 step 稳定性测试

### 数据

新增脚本：

- `scripts/make_synthetic_dpo.py`

生成数据：

```bash
python scripts/make_synthetic_dpo.py --output datasets/synthetic_dpo_256.jsonl --num-rows 256
```

数据文件：

- `datasets/synthetic_dpo_256.jsonl`
- 256 条合成 DPO 样本。
- 格式仍为 `messages` + `rejected_response`。

说明：

- 该数据只用于短程稳定性和吞吐 smoke。
- 不能用于判断真实 DPO 效果。

### 启动方式

在服务器我们的容器中后台启动，并写入日志：

```bash
cd /workspace/llin-rl-dpo
DATASET_PATH=/workspace/llin-rl-dpo/datasets/synthetic_dpo_256.jsonl \
OUTPUT_DIR=/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-stability-20step \
MAX_STEPS=20 \
MASTER_PORT=29619 \
scripts/run_ms_swift_qwen36_dpo_smoke.sh
```

日志文件：

```text
/workspace/llin-rl-dpo/logs/ms_swift_qwen36_dpo_stability_20step.log
```

输出目录：

```text
/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-stability-20step/v0-20260703-120434
```

### 结果

任务退出码：

```text
0
```

训练摘要：

```text
global_step/max_steps=20/20
train_runtime=118.3962
train_samples_per_second=1.351
train_steps_per_second=0.169
train_loss=0.0722111
memory(GiB)=51.93
```

最后一步：

```text
loss=5.117e-05
grad_norm=0.00090253
learning_rate=0.0
rewards/chosen=3.28125
rewards/rejected=-6.96875
rewards/accuracies=1.0
rewards/margins=10.25
logps/chosen=-26.125
logps/rejected=-119.0
train_speed(s/it)=5.91828
```

### 判断

- FSDP2 路线至少能完成 20 step 短程稳定性测试。
- `memory(GiB)` 从 1 step 的约 `51.19` 上升到约 `51.93`，没有出现 OOM。
- 20 step 平均约 `5.92s/step`，比 1 step smoke 的 `139.6s/step` 更接近训练阶段速度；1 step 结果包含明显启动/初始化影响。
- 合成数据上 loss 快速下降、reward margin 增大，只说明链路可优化，不能代表真实效果。

下一步：

1. 引入真实或半真实 DPO 数据，并拆分固定验证集。
2. 跑 100 step 级别，记录 step time 分布、HBM、AICore 利用率。
3. 尝试开启 checkpoint 保存，验证 adapter 保存/恢复。
4. 测试 LLaMA-Factory NPU 的同口径 1 step/20 step。
5. 评估是否安装并测试 `deepspeed zero2/zero3`，作为 FSDP2 对照。

## 2026-07-03 ms-swift DPO/FSDP2 checkpoint 保存与恢复测试

### 脚本更新

`scripts/run_ms_swift_qwen36_dpo_smoke.sh` 从固定 smoke 脚本改成可配置实验入口。

新增环境变量：

- `NUM_TRAIN_EPOCHS`
- `SAVE_STRATEGY`
- `SAVE_STEPS`
- `SAVE_TOTAL_LIMIT`
- `EVAL_STRATEGY`
- `RESUME_FROM_CHECKPOINT`

### checkpoint 保存测试

启动参数：

```bash
DATASET_PATH=/workspace/llin-rl-dpo/datasets/synthetic_dpo_256.jsonl \
OUTPUT_DIR=/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-checkpoint-20step \
MAX_STEPS=20 \
SAVE_STRATEGY=steps \
SAVE_STEPS=10 \
SAVE_TOTAL_LIMIT=2 \
MASTER_PORT=29621 \
scripts/run_ms_swift_qwen36_dpo_smoke.sh
```

输出目录：

```text
/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-checkpoint-20step/v0-20260703-124743
```

结果：

```text
exit_code=0
global_step/max_steps=20/20
train_runtime=97.8253
train_samples_per_second=1.636
train_steps_per_second=0.204
train_loss=0.0722111
memory(GiB)=51.85
```

checkpoint：

```text
checkpoint-10: about 713M
checkpoint-20: about 713M
```

`checkpoint-20` 结构：

```text
optimizer_0/
pytorch_model_fsdp_0/
rng_state_0.pth ... rng_state_7.pth
scheduler.pt
trainer_state.json
```

说明：

- 这是 FSDP2 sharded checkpoint，不是单文件 `adapter_model.safetensors`。
- 保存时出现 HCCL warning：`HCCL doesn't support gather at the moment. Implemented with allgather instead.`，但保存流程完成。

### checkpoint 恢复测试

启动参数：

```bash
DATASET_PATH=/workspace/llin-rl-dpo/datasets/synthetic_dpo_256.jsonl \
OUTPUT_DIR=/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-resume-10-to-12 \
MAX_STEPS=12 \
SAVE_STRATEGY=no \
RESUME_FROM_CHECKPOINT=/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-checkpoint-20step/v0-20260703-124743/checkpoint-10 \
MASTER_PORT=29623 \
scripts/run_ms_swift_qwen36_dpo_smoke.sh
```

结果：

```text
exit_code=1
```

失败点：

```text
TypeError: SwiftModel.load_state_dict() got an unexpected keyword argument 'assign'
```

调用链关键位置：

```text
accelerate.utils.fsdp_utils.fsdp2_load_full_state_dict
model.load_state_dict(sharded_sd, assign=True)
```

判断：

- 当前环境下 `ms-swift + FSDP2 + SwiftModel` 的 checkpoint 保存能成功。
- 当前环境下 FSDP2 checkpoint 恢复不能成功。
- 这会影响正式长训：如果中断后无法 resume，长时间训练风险较高。

下一步：

1. 调查 ms-swift/Accelerate/FSDP2 对 `SwiftModel.load_state_dict(assign=...)` 的兼容修复。
2. 测试是否能导出或保存 LoRA adapter 为普通 `adapter_model.safetensors`，作为最低限度可用产物。
3. 对比 `--fsdp fsdp2` 与其他后端，例如 deepspeed zero2/zero3 或 FSDP1，检查保存/恢复行为。
4. 在 checkpoint 恢复解决前，不建议直接启动长时间正式训练。

## 2026-07-03 resume 继续排查与 adapter 导出路线

### `assign` 兼容 patch

新增文件：

- `patches/sitecustomize.py`

启用方式：

```bash
LLIN_SWIFTMODEL_ASSIGN_PATCH=1
```

脚本行为：

- `scripts/run_ms_swift_qwen36_dpo_smoke.sh` 在该变量为 `1` 时，将 `/workspace/llin-rl-dpo/patches` 加入 `PYTHONPATH`。
- Python 启动时自动加载 `sitecustomize.py`。
- patch 让 `SwiftModel.load_state_dict` 接受 `assign=` 参数。

### resume patch 测试

启动参数：

```bash
LLIN_SWIFTMODEL_ASSIGN_PATCH=1 \
DATASET_PATH=/workspace/llin-rl-dpo/datasets/synthetic_dpo_256.jsonl \
OUTPUT_DIR=/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-resume-patch-10-to-12 \
MAX_STEPS=12 \
SAVE_STRATEGY=no \
RESUME_FROM_CHECKPOINT=/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-checkpoint-20step/v0-20260703-124743/checkpoint-10 \
MASTER_PORT=29625 \
scripts/run_ms_swift_qwen36_dpo_smoke.sh
```

结果：

```text
exit_code=1
```

进展：

- 不再报 `SwiftModel.load_state_dict() got an unexpected keyword argument 'assign'`。

新的失败点：

```text
RuntimeError: Missing key in checkpoint state_dict: model.model.visual.patch_embed.proj.weight.
```

判断：

- 默认 FSDP2 sharded checkpoint 中没有完整视觉塔底座权重。
- FSDP2 resume loader 期待完整模型 sharded state。
- `assign` patch 只解决了第一层签名问题，不能让默认 sharded checkpoint 直接恢复。

### FULL_STATE_DICT + save_only_model 导出测试

新增配置：

- `configs/fsdp2_full_state.json`

关键内容：

```json
{
  "fsdp": "full_shard auto_wrap",
  "fsdp_config": {
    "fsdp_version": 2,
    "reshard_after_forward": true,
    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    "cpu_ram_efficient_loading": true,
    "state_dict_type": "FULL_STATE_DICT",
    "activation_checkpointing": true
  }
}
```

启动参数：

```bash
DATASET_PATH=/workspace/llin-rl-dpo/datasets/synthetic_dpo_256.jsonl \
OUTPUT_DIR=/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step \
MAX_STEPS=2 \
SAVE_STRATEGY=steps \
SAVE_STEPS=1 \
SAVE_TOTAL_LIMIT=1 \
SAVE_ONLY_MODEL=true \
FSDP_CONFIG=/workspace/llin-rl-dpo/configs/fsdp2_full_state.json \
MASTER_PORT=29627 \
scripts/run_ms_swift_qwen36_dpo_smoke.sh
```

结果：

```text
exit_code=0
global_step/max_steps=2/2
train_runtime=29.94
train_samples_per_second=0.534
train_steps_per_second=0.067
train_loss=0.5927
memory(GiB)=51.19
```

输出目录：

```text
/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step/v0-20260703-130918/checkpoint-2
```

产物：

```text
adapter_config.json
adapter_model.safetensors
additional_config.json
args.json
trainer_state.json
training_args.bin
```

大小：

```text
checkpoint-2: 223M
```

adapter 检查：

```text
adapter_type=LORA
base_model=/models/Qwen3.6-27B
num_tensors=992
first_key=base_model.model.model.language_model.layers.0.linear_attn.in_proj_a.lora_A.weight
last_key=base_model.model.model.language_model.layers.9.mlp.up_proj.lora_B.weight
```

结论：

- 当前 sharded checkpoint resume 仍未通过。
- 但 `FULL_STATE_DICT + save_only_model=true` 可以导出普通 LoRA adapter。
- 这是当前最实用的产物保障路线：长训前仍需解决 resume，但短程/阶段性训练至少可以保存 adapter 产物。

下一步：

1. 测试该 LoRA adapter 是否能重新加载做最小推理。
2. 继续调查 FSDP2 sharded checkpoint resume 的根修复：是需要完整 state_dict、忽略冻结视觉塔缺失 key，还是 ms-swift/Accelerate 版本差异。
3. 与 deepspeed zero2/zero3 保存/恢复对照。

## 2026-07-03 adapter 重新加载推理 smoke test

目的：

- 验证 `FULL_STATE_DICT + save_only_model=true` 导出的普通 LoRA adapter 不是只能被 safetensors 读取，而是能被 ms-swift 重新挂载到 `/models/Qwen3.6-27B` 并执行一次非交互推理。

新增文件：

```text
datasets/tiny_infer.jsonl
scripts/run_ms_swift_adapter_infer_smoke.sh
```

默认 adapter：

```text
/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step/v0-20260703-130918/checkpoint-2
```

启动命令：

```bash
cd /workspace/llin-rl-dpo
scripts/run_ms_swift_adapter_infer_smoke.sh
```

脚本核心参数：

```text
swift infer
--model /models/Qwen3.6-27B
--model_type qwen3_5
--adapters /workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step/v0-20260703-130918/checkpoint-2
--load_args false
--infer_backend pt
--device_map auto
--torch_dtype bfloat16
--max_new_tokens 16
--stream false
--val_dataset /workspace/llin-rl-dpo/datasets/tiny_infer.jsonl
--val_dataset_sample 1
```

结果：

```text
exit_code=0
result_path=/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step/v0-20260703-130918/checkpoint-2/infer_result/20260703-132529.jsonl
num_prompt_tokens=37
num_generated_tokens=16
runtime=49.29208978707902
samples/s=0.020287230756893794
tokens/s=0.3245956921103007
```

输出样例：

```json
{"response":"<think>\n\n</think>\n\nIt prevents wasting significant compute resources on a run that would fail due to corrupted or", "labels": null}
```

注意：

- `MAX_NEW_TOKENS=16` 太短，输出被截断；本测试只证明 adapter 重新加载和推理链路可用，不代表模型效果。
- 推理日志出现 `chunk_gated_delta_rule` warning：
  - `Input tensor shape suggests format mismatch: seq_len (37) < num_heads (48).`
- 命令仍正常完成；后续需要单独确认该 warning 是否影响长上下文、效率或正确性。

当前结论：

- ms-swift 目前不只是能训练 1/20 step，也能导出普通 LoRA adapter，并完成 adapter 重新加载推理 smoke test。
- 默认 FSDP2 sharded checkpoint resume 仍未通过；当前可用的产物保障路线是 `FULL_STATE_DICT + save_only_model=true` 导出 adapter。

下一步：

1. 做真实 DPO 样例集或小规模业务数据集的 100-500 step 试跑。
2. 增加固定 prompts 的 before/after adapter 对照推理，评估输出方向而不是只看 loss。
3. 继续对照 deepspeed zero2/zero3 保存、resume 和 adapter 导出行为。

## 2026-07-03 固定 prompts 的 base/adapter 对照推理

目的：

- 使用同一组固定 prompts 对比 base 模型和上一轮 2 step 合成 DPO LoRA adapter 的输出与推理速度。
- 这不是正式效果评测，只是确认 adapter 重新加载后的输出方向和开销。

新增文件：

```text
datasets/fixed_eval_prompts.jsonl
scripts/run_ms_swift_fixed_prompt_eval.sh
scripts/run_ms_swift_base_adapter_compare.sh
```

固定 prompts：

```text
1. List two checks we should run before starting a long DPO training job.
2. In one paragraph, explain why saving only an adapter may be safer than relying on an untested sharded checkpoint.
```

实测命令等价于：

```bash
MAX_NEW_TOKENS=64 \
RESULT_PATH=/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/base-20260703.jsonl \
scripts/run_ms_swift_fixed_prompt_eval.sh

ADAPTER_PATH=/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-fullstate-saveonly-2step/v0-20260703-130918/checkpoint-2 \
MAX_NEW_TOKENS=64 \
RESULT_PATH=/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/adapter-20260703.jsonl \
scripts/run_ms_swift_fixed_prompt_eval.sh
```

base 结果：

```text
result_path=/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/base-20260703.jsonl
num_prompt_tokens=85
num_generated_tokens=128
num_samples=2
runtime=27.873314147931524
samples/s=0.07175321848652216
tokens/s=4.5922059831374185
chunk_gated_delta_rule_warning_count=2
```

adapter 结果：

```text
result_path=/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/adapter-20260703.jsonl
num_prompt_tokens=85
num_generated_tokens=128
num_samples=2
runtime=42.14306939195376
samples/s=0.047457388103341464
tokens/s=3.0372728386138537
chunk_gated_delta_rule_warning_count=2
```

观察：

- base 和 adapter 输出方向基本一致。
- adapter 第一条输出中 `reward(chosen)` 被写成 `r(chosen)`，并继续生成第二点，但 64 token 上限导致末尾截断。
- 第二条输出两边几乎一致，也在末尾截断。
- adapter 推理速度低于 base：本次小样本观测为 `3.0373` vs `4.5922 tokens/s`。
- `chunk_gated_delta_rule` tensor shape warning 在 base 和 adapter 都出现各 2 次，因此不是 adapter 独有现象。

结论：

- 2 step 合成 DPO adapter 太小，不能期待明显效果变化。
- 这次对照说明 adapter 可复现加载并产生合理方向输出，但还不能证明真实 DPO 效果。
- 正式效果评测需要：
  - 更长 token 上限或禁用 thinking，避免输出截断。
  - 固定真实验证集。
  - base vs adapter 自动打分或人工 win-rate。

## 2026-07-03 non-thinking 128 token 固定 prompts 对照

目的：

- 在上一轮固定 prompts 对照基础上，提高输出上限到 `128`，并传入 `ENABLE_THINKING=false`。
- 检查是否能去掉 thinking 内容、减少截断，并获得更稳定的 base/adapter 输出对比。

脚本更新：

```text
scripts/run_ms_swift_fixed_prompt_eval.sh
scripts/run_ms_swift_base_adapter_compare.sh
```

新增支持的环境变量：

```text
ENABLE_THINKING
PRESERVE_THINKING
TEMPERATURE
```

启动命令：

```bash
RUN_ID=nonthinking-128-20260703 \
MAX_NEW_TOKENS=128 \
ENABLE_THINKING=false \
scripts/run_ms_swift_base_adapter_compare.sh
```

确认实际命令包含：

```text
--enable_thinking false
```

base 结果：

```text
exit_code=0
result_path=/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/base-nonthinking-128-20260703.jsonl
num_prompt_tokens=85
num_generated_tokens=256
num_samples=2
runtime=53.788713496993296
samples/s=0.037182521573262704
tokens/s=4.759362761377626
chunk_gated_delta_rule_warning_count=2
think_prefix_count=2
```

adapter 结果：

```text
exit_code=0
result_path=/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/adapter-nonthinking-128-20260703.jsonl
num_prompt_tokens=85
num_generated_tokens=240
num_samples=2
runtime=77.89536395308096
samples/s=0.025675468968919233
tokens/s=3.081056276270308
chunk_gated_delta_rule_warning_count=2
think_prefix_count=2
```

观察：

- `ENABLE_THINKING=false` 确实传入了 ms-swift。
- 输出没有展开思考内容，但仍保留空的 `<think></think>` 前缀；这应视为模板行为。
- base 两条输出仍在 128 token 上限处截断。
- adapter 两条输出均完整结束，内容方向合理：
  - 第一条给出 reward consistency 和 data pipeline stress-test。
  - 第二条解释 adapter 比未验证 sharded checkpoint 更容易验证和回滚。
- adapter 推理仍慢于 base，本轮观测为 `3.0811` vs `4.7594 tokens/s`。

结论：

- 对自动评测而言，不能仅依赖 `enable_thinking=false` 获得完全干净文本；还需要后处理移除空 `<think></think>` 前缀，或继续研究 template 参数。
- 2 step 合成 adapter 仍只能说明链路，不说明真实效果。
- 下一步应转向真实/半真实 DPO 小样本 100-500 step，并用本固定 prompts 流程做 base/adapter 对照。

## 2026-07-03 半真实 ops DPO 512 数据与 100 step 试跑

目的：

- 不再继续打磨 2 step 玩具 adapter。
- 构造一个更贴近当前任务域的半真实 DPO 数据集，覆盖训练运维、checkpoint、resume、共享服务器安全、框架评估、adapter 导出、实验记录和网络约束。
- 使用已验证的 `FULL_STATE_DICT + save_only_model=true` 路线跑 100 step，并验证 adapter 导出和重新加载推理。

新增文件：

```text
scripts/make_ops_dpo.py
datasets/ops_dpo_512.jsonl
scripts/run_ms_swift_ops_dpo_100step.sh
```

数据：

```text
rows=512
source=templated_ops_dpo_v1
format=messages + rejected_response
```

样例主题：

```text
long DPO run readiness
checkpoint policy
container safety
framework comparison
adapter export
experiment logging
network limits
effect evaluation
```

训练命令：

```bash
RUN_ID=ops-100step-20260703 \
scripts/run_ms_swift_ops_dpo_100step.sh
```

脚本实际配置：

```text
DATASET_PATH=/workspace/llin-rl-dpo/datasets/ops_dpo_512.jsonl
OUTPUT_DIR=/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-ops-100step/ops-100step-20260703
MAX_STEPS=100
NUM_TRAIN_EPOCHS=3
SAVE_STRATEGY=steps
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=1
SAVE_ONLY_MODEL=true
FSDP_CONFIG=/workspace/llin-rl-dpo/configs/fsdp2_full_state.json
MASTER_PORT=29643
```

训练结果：

```text
exit_code=0
output_dir=/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-ops-100step/ops-100step-20260703/v0-20260703-141605
global_step/max_steps=100/100
train_runtime=344.9065
train_samples_per_second=2.319
train_steps_per_second=0.29
train_loss=0.01584221
memory(GiB)=52.56
```

最终 step：

```text
loss=1.607e-07
learning_rate=0.0
rewards/chosen=7.375
rewards/rejected=-10.625
rewards/accuracies=1.0
rewards/margins=18.0
epoch=1.5625
```

adapter 产物：

```text
/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-ops-100step/ops-100step-20260703/v0-20260703-141605/checkpoint-100
```

adapter 检查：

```text
adapter_model.safetensors: about 223M
adapter_type=LORA
base_model=/models/Qwen3.6-27B
num_tensors=992
first_key=base_model.model.model.language_model.layers.0.linear_attn.in_proj_a.lora_A.weight
last_key=base_model.model.model.language_model.layers.9.mlp.up_proj.lora_B.weight
```

观察：

- 训练顺利完成，100 step DPO + FSDP2 + LoRA + 8 NPU 链路通过。
- `FULL_STATE_DICT + save_only_model=true` 再次成功导出普通 LoRA adapter。
- 半真实模板数据非常容易拟合，step 5 后 loss 已快速降低，后续 reward accuracy 基本为 `1.0`。
- 这说明训练链路可用，但不能单独证明模型泛化效果。

## 2026-07-03 ops 100-step adapter 固定 prompts 对照

目的：

- 验证 100 step adapter 可以重新加载推理。
- 使用同一套 fixed prompts 对比 base 和 100-step adapter 输出。

启动命令：

```bash
RUN_ID=ops-100step-eval-20260703 \
MAX_NEW_TOKENS=128 \
ENABLE_THINKING=false \
ADAPTER_PATH=/workspace/llin-rl-dpo/outputs/ms-swift-qwen36-dpo-ops-100step/ops-100step-20260703/v0-20260703-141605/checkpoint-100 \
scripts/run_ms_swift_base_adapter_compare.sh
```

base 结果：

```text
exit_code=0
result_path=/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/base-ops-100step-eval-20260703.jsonl
num_prompt_tokens=85
num_generated_tokens=194
num_samples=2
runtime=63.244178544031456
samples/s=0.03162346394629148
tokens/s=3.0674760027902734
chunk_gated_delta_rule_warning_count=2
think_prefix_count=2
```

adapter 结果：

```text
exit_code=0
result_path=/workspace/llin-rl-dpo/outputs/ms-swift-fixed-prompt-eval/adapter-ops-100step-eval-20260703.jsonl
num_prompt_tokens=85
num_generated_tokens=194
num_samples=2
runtime=63.95430243492592
samples/s=0.03127232920779674
tokens/s=3.0334159331562836
chunk_gated_delta_rule_warning_count=2
think_prefix_count=2
```

输出对比：

- 两条 fixed prompts 上，base 和 adapter 输出完全一致。
- 两边仍有空 `<think></think>` 前缀。
- 两边各出现 2 次 `chunk_gated_delta_rule` warning。

结论：

- 100 step adapter 重新加载推理通过。
- 这组 fixed prompts 没有观察到 base/adapter 输出差异。
- 由于训练数据是模板化半真实数据，这个结果只能说明：
  - ms-swift 训练/保存/加载链路通过。
  - adapter 对这 2 条固定 prompts 没有可见影响。
  - 效果评估需要真实验证集或更贴近训练目标的 held-out prompts。
## 2026-07-06 MindSpeed-MM Qwen3.6-27B SFT cutoff 复现

目标：复查老板反馈的 MindSpeed-MM SFT 在 `cutoff > 2048` 时可能触发 `aclnnRotaryPositionEmbeddingGrad error 561002`。

约束：

- 不改宿主机。
- 不动别人的 Docker。
- 优先使用我们自己的容器和工作区。
- 服务器只能稳定访问部分中国大陆网站，依赖安装优先使用大陆镜像。

准备过程：

1. 在我们自己的容器/工作区中准备 MindSpeed-MM 参考源码：`reference/MindSpeed-MM`。
2. 补齐配套 MindSpeed：`reference/MindSpeed-26.0.0_core_r0.12.1`，对应 MindSpeed-MM 26.0.0 文档要求。
3. 使用 `checkpoint.convert_cli Qwen35Converter hf_to_dcp` 将 `/models/Qwen3.6-27B` 转换为 DCP 权重，输出到 `/workspace/llin-rl-dpo/checkpoints/msmm-qwen36-27b-dcp`。
4. 生成 8 条长样本 SFT probe 数据：`datasets/msmm_sft_probe_long.jsonl`。
5. 生成两份 1 step 配置：`configs/msmm_qwen36_sft_cutoff2048_1step.yaml` 和 `configs/msmm_qwen36_sft_cutoff4096_1step.yaml`。
6. 新建多个 MindSpeed-MM 探测容器时发现 NPU runtime 映射不稳定：部分新容器即使设备节点存在，`torch.npu.device_count()` 仍为 `0`。
7. 最终采用稳定可见 8 NPU 的我们自己的 `llin-rl-dpo` 容器，并创建隔离 venv：`/workspace/llin-rl-dpo/.venvs/msmm-qwen36`。

关键环境：

- `torch 2.7.1+cpu`
- `torch_npu 2.7.1.post4`
- `transformers 5.2.0`
- `accelerate 1.2.0`
- `datasets 5.0.0`
- `triton-ascend 3.2.1`
- `ASCEND_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`

中间失败与修复：

- `transformers 5.13.0` 失败于 `create_causal_mask() got an unexpected keyword argument 'cache_position'`，按 MindSpeed-MM Qwen3.6 README 降到 `transformers==5.2.0` 后解决。
- 4 NPU 能进入初始化但 Qwen3.6-27B SFT OOM；官方 `examples/qwen3_6/finetune_qwen3_6_27B.sh` 默认 `NPUS_PER_NODE=16`，因此 4 NPU OOM 不代表 rotary 失败。
- HCCL 默认 NPU socket 端口可能冲突，复现实验中显式设置 `HCCL_NPU_SOCKET_PORT_RANGE`。

结果：

- `cutoff=2048`：
  - 日志：`logs/msmm_qwen36_sft_cutoff2048_1step_8npu_venv_20260706.log`
  - exit code：`0`
  - `iteration 1/1`
  - `elapsed time per iteration (ms): 125604.0`
  - `global batch size: 8`
  - `loss: 1.220408E+01`
  - checkpoint：`outputs/msmm-qwen36-sft-cutoff2048-1step/iter_0000001`
  - 未出现 `RotaryPositionEmbeddingGrad` 或 `561002`

- `cutoff=4096`：
  - 日志：`logs/msmm_qwen36_sft_cutoff4096_1step_8npu_venv_20260706.log`
  - exit code：`0`
  - `iteration 1/1`
  - `elapsed time per iteration (ms): 53971.2`
  - `global batch size: 8`
  - `loss: 1.006310E+01`
  - checkpoint：`outputs/msmm-qwen36-sft-cutoff4096-1step/iter_0000001`
  - 未出现 `RotaryPositionEmbeddingGrad` 或 `561002`

当前判断：

- 在本次环境、数据和 8 NPU 1 step SFT 复现条件下，`cutoff=4096` 没有复现老板截图中的 `aclnnRotaryPositionEmbeddingGrad error 561002`。
- 这不能证明所有 MindSpeed-MM SFT 场景都不会触发该错误；差异可能来自 MindSpeed-MM/MindSpeed/torch_npu/CANN 版本、真实数据长度分布、是否走 fused/fast path、NPU 数量和容器映射方式。
- 如果要进一步逼近老板的结果，下一步应复用他的 MindSpeed-MM commit、CANN/torch_npu 版本、完整真实 SFT 数据和原始启动参数，再跑 `cutoff=4096/8192/16384`。

## 2026-07-06 MindSpeed-MM cutoff=4096 2-step 与 long-answer 复查

目的：

- 回答“是不是我们刚刚的训练数据其实没有到 4096”的问题。
- 对照老板截图中 iteration 2 才 OOM 的现象，补充 `cutoff=4096` 的 2-step 测试。
- 构造 long-answer 样本，确认 answer loss 覆盖到 4096 附近，而不是只有 prompt 被截断到 4096。

运行环境：

- 容器：`llin-rl-dpo`
- venv：`/workspace/llin-rl-dpo/.venvs/msmm-qwen36`
- 设备：`ASCEND_VISIBLE_DEVICES=8,9,10,11,12,13,14,15`
- 逻辑 NPU：8
- 权重：`/workspace/llin-rl-dpo/checkpoints/msmm-qwen36-27b-dcp`
- 关键配置保持：
  - `freeze: model.visual`
  - `recompute: true`
  - `enable_chunk_loss: true`
  - `chunk_size: 1024`
  - `micro_batch_size: 1`
  - `gradient_accumulation_steps: 1`

短 answer 2-step：

```text
config=/workspace/llin-rl-dpo/configs/msmm_qwen36_sft_cutoff4096_2step.yaml
log=/data/liulin/llin-rl-dpo/logs/msmm_qwen36_sft_cutoff4096_2step_8npu_venv_20260706.log
exit_code=0
iteration 1/2 elapsed=16419.2 ms loss=1.006310E+01 grad_norm=505.511
iteration 2/2 elapsed=14716.3 ms loss=2.602338E+00 grad_norm=90.934
allocated=38700.54248046875 MB
max_allocated=58801.4921875 MB
reserved=58880.0 MB
max_reserved=61132.0 MB
```

结果：

- 未出现 `aclnnRotaryPositionEmbeddingGrad`。
- 未出现 `561002`。
- 未出现 `aclnnCat alloc device memory failed`。
- checkpoint 保存到 `/workspace/llin-rl-dpo/outputs/msmm-qwen36-sft-cutoff4096-2step/iter_0000002`。

long-answer 2-step：

```text
dataset=/workspace/llin-rl-dpo/datasets/msmm_sft_probe_long_answer.jsonl
config=/workspace/llin-rl-dpo/configs/msmm_qwen36_sft_cutoff4096_longanswer_2step.yaml
log=/data/liulin/llin-rl-dpo/logs/msmm_qwen36_sft_cutoff4096_longanswer_2step_8npu_venv_20260706.log
exit_code=0
iteration 1/2 elapsed=16957.6 ms loss=1.215048E+01 grad_norm=3243.944
iteration 2/2 elapsed=14105.7 ms loss=5.574024E-02 grad_norm=3.186
allocated=38700.54248046875 MB
max_allocated=58801.4921875 MB
reserved=58880.0 MB
max_reserved=61132.0 MB
```

结果：

- 未出现 `aclnnRotaryPositionEmbeddingGrad`。
- 未出现 `561002`。
- 未出现 `aclnnCat alloc device memory failed`。
- checkpoint 保存到 `/workspace/llin-rl-dpo/outputs/msmm-qwen36-sft-cutoff4096-longanswer-2step/iter_0000002`。

long-answer cache 统计：

```text
cache=/workspace/llin-rl-dpo/cache/msmm_sft_cutoff4096_longanswer_2step
rows=8
input_ids length min/max=4096/4096
labels != -100 min/max=4073/4073
first supervised label position=23
last supervised label position=4095
```

判断：

- 这次 long-answer 测试可以排除“我们的训练没有真正到 4096”这一解释。
- 在当前 MindSpeed-MM 路径、当前配置、8 NPU 和无图像输入条件下，`cutoff=4096` 的 forward/backward、第二步 optimizer 前后、checkpoint 保存都能通过。
- 但这仍不等同于老板截图里的全参数/patch 场景。截图中提到的 torch 原生 rotary patch 可能走 `torch.cat([q_embed, q_pass], dim=-1)`，会额外分配 tensor；而本次 max reserved 已达到约 `61.1GB/64GB`，显存余量很薄。
- 因此，老板截图中 `aclnnCat alloc device memory failed` 与我们本轮通过并不矛盾：它更像是 patch 绕过 rotary 算子限制后引入的显存压力问题，而不是同一个 `aclnnRotaryPositionEmbeddingGrad error 561002` 问题。

下一步建议：

1. 如果目标是复现老板截图，需获取其原始启动参数、是否启用 torch 原生 rotary patch、是否全参数训练、是否含图像输入、是否冻结视觉塔、是否开启 recompute/chunk loss。
2. 在确认真实配置前，不建议把当前结果表述为“4096 全参数一定没问题”；更准确说法是：当前隔离环境下，MindSpeed-MM Qwen3.6-27B SFT `cutoff=4096` 在 8 NPU、冻结视觉塔、重计算和 chunk loss 条件下可跑通 2 step。
3. 若必须跑全参数 4096，优先考虑更多卡分片、保持 fused rotary、开启/加大重计算、降低 micro batch 或改 LoRA；否则很容易越过 64GB HBM 边界。
