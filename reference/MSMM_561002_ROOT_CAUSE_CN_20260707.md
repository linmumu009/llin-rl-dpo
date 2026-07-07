# MindSpeed-MM Qwen3.6 561002 根因分析 - 2026-07-07

## 结论先行

这次 `aclnnRotaryPositionEmbeddingGrad error 561002` 的直接失败点已经可以收敛到：

```text
Qwen3.6 text attention
  -> apply_rotary_pos_emb
  -> torch_npu.npu_rotary_mul
  -> backward
  -> aclnnRotaryPositionEmbeddingGrad
  -> RopeHalfGradSetTiling
  -> reserveAlignNum = 2592 too large
```

它不是数据格式不支持、不是 latest MindSpeed-MM 不支持 Qwen3.6、不是视觉塔、不是 activation offload，也不是 FSDP reduce-scatter 本身。它是 Ascend NPU fused rotary backward 算子在 MindSpeed-MM Qwen3.6 训练路径里的一个 tiling 失败。

更准确地说：问题不是“只要 `cutoff_len=4096` 就必然失败”，也不是“只要序列长度超过 2048 就必然失败”。触发点是老板数据中的长 `reasoning_content` 经 OpenAI converter 合并后，进入第 2 个 global batch 的 rank4，本地 batch 被 pad 到 `2576`，在 Qwen3.6 的 partial rotary 维度 `64` 上调用 NPU fused rotary backward，最终 CANN/op-plugin 内部得到 `reserveAlignNum=2592` 并报 `561002`。

## 本次新增证据

本次使用的是非 DPO SFT 实验容器：

```text
llin-msmm-sft-latest-run
```

没有进入或修改 DPO 容器。latest 复现环境为：

```text
/workspace/MindSpeed-MM-latest  commit 643738f
/workspace/MindSpeed-latest     commit 38ecf80
/workspace/msmm-latest-probe-venv
torch_npu 2.7.1.post4
transformers 5.2.0
triton-ascend 3.2.1
CANN /usr/local/Ascend/cann-9.0.0
```

训练配置对齐老板最小复现：

```text
数据：老板 OpenAI 格式数据前 32 条
template：qwen3_6
cutoff_len：4096
micro_batch_size：2
global_batch_size：16
8 NPU
Full SFT，freeze: []
activation offload: true
chunk loss: 256
train_iters: 2
```

latest MindSpeed-MM 支持 `formatting: openai` 和 `template: qwen3_6`，但第二步仍在 rank4 复现同一个错误：

```text
RuntimeError: npu_rotary_mul_backward:
NPU function error: call aclnnRotaryPositionEmbeddingGrad failed, error code is 561002
reserveAlignNum = 2592 too large, aicore do not support.
```

## 数据如何触发

最开始容易误判的一点是：OpenAI 数据里 assistant 的 `content` 看起来很短。但 MindSpeed-MM latest 的 `OpenAIDatasetConverter` 会把 assistant 的 `reasoning_content` 合并进去：

```python
<think>
reasoning_content
</think>

content
```

所以真实训练长度必须把 `reasoning_content` 算进去。

按真实 tokenizer `/models/Qwen3.6-27B`、真实模板 `qwen3_6`、真实 `infer_seqlen` 计算，前 32 条里关键长度为：

```text
row20: total 2506, source 454, target/label 2052, reasoning_content 3776 chars
row28: total 1264, source 557, target/label 707,  reasoning_content 1578 chars
row31: total 3506, source 461, target/label 3045, reasoning_content 6386 chars
```

sampler 配置是：

```text
shuffle: false
world_size: 8
micro_batch_size: 2
```

因此前 32 条的 rank-local 分配是：

```text
iteration 1 rank4: rows 4, 12
iteration 2 rank4: rows 20, 28
```

这和失败日志的 rank4、iteration 2 对上。`row20` 是 rank4 第二步的主触发样本。

## 真实 rotary 输入 shape

为了避免只靠推断，我们在 SFT 实验容器里加了一个临时 shape probe wrapper：

```text
/workspace/MindSpeed-MM-latest/llin_shape_probe_trainer.py
/workspace/MindSpeed-MM-latest/run_llin_latest_rjx_rows0_31_2step_shapeprobe.sh
```

这个 wrapper 只包一层 `torch_npu.npu_rotary_mul` 打印 shape/stride/dtype/device，然后调用原函数；没有替换 rotary 算法。

shape probe 日志：

```text
/workspace/MindSpeed-MM-latest/llin_logs/latest_rjx_rows0_31_shapeprobe_8npu_20260707_081147.log
```

iteration 1 通过时，rank4 的文本 rotary 输入为：

```text
q_shape (2, 24, 1312, 64)
q_shape (2, 4, 1312, 64)
```

iteration 2 失败前，rank4 的文本 rotary 输入变为：

```text
q_shape   (2, 24, 2576, 64)
q_stride  (15826944, 256, 6144, 1)
q_contig  False
cos_shape (2, 1, 2576, 64)
dtype     bfloat16
device    npu:4

q_shape   (2, 4, 2576, 64)
q_stride  (2637824, 256, 1024, 1)
q_contig  False
cos_shape (2, 1, 2576, 64)
dtype     bfloat16
device    npu:4
```

随后同步报错：

```text
npu_rotary_mul_backward
aclnnRotaryPositionEmbeddingGrad failed, error code is 561002
reserveAlignNum = 2592 too large
```

这里的 `2576` 来自 collator 对 rank4 第二步本地 batch 的 padding。row20 原始 tokenized 长度是 `2506`，row28 是 `1264`，batch pad 到 `2576` 后进入 rotary。CANN tiling 内部再得到 `reserveAlignNum=2592`。

## 为什么不是简单的 shape 数字问题

我们单独构造了和失败前完全相同的 shape，直接调用 `torch_npu.npu_rotary_mul` backward：

```text
(2, 24, 2576, 64)
(2, 4, 2576, 64)
```

并测试 contiguous 与非 contiguous 两种 q tensor，结果都通过：

```text
OK exact H 24 contig False q_shape (2, 24, 2576, 64)
OK exact H 24 contig True  q_shape (2, 24, 2576, 64)
OK exact H 4  contig False q_shape (2, 4, 2576, 64)
OK exact H 4  contig True  q_shape (2, 4, 2576, 64)
```

也就是说，不能把根因简化成“这个 shape 一定挂”。更合理的判断是：

```text
完整 MindSpeed-MM 训练路径
  + Qwen3.6 mixed attention / partial rotary
  + FSDP full shard
  + recompute / activation offload
  + NPU tensor internal format / autograd 保存状态
  + rank4 第二步 padded seq_len 2576
共同进入 aclnnRotaryPositionEmbeddingGrad 后触发 tiling 缺陷。
```

我们能确定第一失败算子是 `aclnnRotaryPositionEmbeddingGrad`。但仅凭 Python 层 shape，还不能完全还原 CANN tiling 内部为什么把该场景算成 `reserveAlignNum=2592`。

## 已排除的解释

1. 不是 latest MindSpeed-MM 不支持 Qwen3.6/OpenAI。

latest 源码可以加载 `OpenAIDatasetConverter`、`template=qwen3_6`，对应单测通过；训练也真正走到了 forward/backward。

2. 不是 DPO 容器或 DPO 配置污染。

本次 latest 训练在 `llin-msmm-sft-latest-run` 中完成，没有进入或修改 DPO 容器。

3. 不是视觉塔导致。

数据是纯文本，`images: null`；失败算子在 text attention rotary backward。

4. 不是 activation offload 的必要条件。

此前关闭 activation offload 仍先复现同一个 rotary 561002。

5. 不是 FSDP reduce-scatter 本身。

开启 `ASCEND_LAUNCH_BLOCKING=1` 后，同步栈明确指向：

```text
loss.backward()
  -> npu_rotary_mul_backward
  -> aclnnRotaryPositionEmbeddingGrad
```

6. 不是“所有 4096 都失败”。

之前 4096 long-answer smoke 能通过；本次失败 batch 的实际 padded rotary seq_len 是 `2576`，不是 `4096`。

7. 不是单纯 Python 包版本差异。

旧 MindSpeed-MM 路径没有复现第二步 rotary 错，但旧源码不能等价运行 `formatting: openai + template: qwen3_6`，只能转成 `sharegpt + qwen3_vl`。这会把 row20 从 `2506/2052` 变成 `475/21`，本质上改变了样本长度和模板路径。

## 对 cutoff=2048 的解释

`cutoff_len=2048` 能通过，是因为 row20 被截短：

```text
row20 cutoff=4096: total 2506, label 2052
row20 cutoff=2048: total 2048, label 1594
```

这避开了本次危险的 padded rotary shape。但对沙箱轨迹这类长 CoT 数据，`cutoff=2048` 会严重截断监督信号，所以它是规避方案，不是根治方案。

## 对 torch 原生 rotary patch OOM 的解释

老板截图里提到的另一条路线是把 NPU fused rotary 换成 torch 原生 rotary。这个方向能绕过 `aclnnRotaryPositionEmbeddingGrad`，但会引入更多中间 tensor，例如 `torch.cat`，显存压力更大。

我们之前在 8 卡全参 `cutoff=4096` 场景下已经看到显存边界非常紧，去掉视觉塔冻结后会 OOM。因此：

```text
fused rotary: 省显存，但在这个场景触发 561002
torch 原生 rotary: 可能绕过 561002，但更容易 OOM
```

这两个现象并不矛盾。

## 当前最可信的根因表述

建议对外或对老板这样表述：

```text
我们已经在老板数据和 latest MindSpeed-MM/Qwen3.6/OpenAI 路径上复现了 561002。问题出在 Qwen3.6 text attention 的 NPU fused rotary backward：rank4 第二步处理 row20/row28 时，OpenAI reasoning_content 被合并成长监督样本，batch padding 后 rotary 输入 seq_len 为 2576，partial rotary dim 为 64，进入 aclnnRotaryPositionEmbeddingGrad 后 CANN tiling 计算出 reserveAlignNum=2592，超过该算子当前 aicore 支持范围，于是报 561002。

它不是数据格式、视觉塔、activation offload 或 FSDP reduce-scatter 的表层问题。单独构造相同 shape 调 npu_rotary_mul backward 可以通过，说明还依赖完整训练路径里的 NPU 内部格式/重计算/FSDP/autograd 状态；但第一失败算子和触发 batch 已经定位清楚。
```

## 后续可选方向

1. 最务实的训练规避：`cutoff_len <= 2048` 或过滤/分桶长 `reasoning_content`。

缺点是会截断长 CoT，和“不截断沙箱轨迹”的目标冲突。

2. 更接近目标的规避：保留 `cutoff=4096`，但对 rotary 做 fallback。

可以只在危险 shape 上从 `torch_npu.npu_rotary_mul` fallback 到 torch 实现，避免所有 batch 都吃额外显存。风险是代码复杂，并且仍可能 OOM。

3. 显存配套策略：如果使用 torch rotary fallback，需要同时降低显存压力。

可组合：

```text
freeze visual
micro_batch_size=1
更多卡，例如 16 卡
更强 recompute
LoRA / 冻结部分模块
减少 optimizer 状态
```

4. 框架/算子上报：向 MindSpeed-MM / torch-npu / CANN 侧提交最小复现。

最小复现应包含：

```text
模型：Qwen3.6-27B
数据：前 32 条 OpenAI 数据，保留 reasoning_content
配置：8 卡、Full SFT、cutoff=4096、mbs=2、shuffle=false
触发：iteration 2 rank4
shape：q_rot (2,24,2576,64) 和 (2,4,2576,64)
错误：reserveAlignNum=2592
```

5. 更深入的下一步实验：做“只 fallback rank4/seq_len=2576 的 rotary backward”。

如果能在同配置下跑过 iteration 2，就能进一步证明：训练链路本身没有逻辑错误，唯一 blocker 是 fused rotary backward 算子。这个实验要小心显存，最好先冻结视觉塔或改 `micro_batch_size=1`。
