# MindSpeed-MM latest 561002 repro - 2026-07-07

## Purpose

Test whether latest MindSpeed-MM fixes the previously reproduced Qwen3.6-27B SFT `aclnnRotaryPositionEmbeddingGrad error 561002`.

This experiment uses the latest Qwen3.6/OpenAI data path:

- `formatting: openai`
- `template: qwen3_6`
- OpenAI ChatCompletion-style boss data

## Safety boundary

- Did not use or modify the DPO container `llin-rl-dpo`.
- Stopped old idle `llin-msmm-*` and `llin-autoresearch*` experiment containers after user approval, because they held Ascend devices even with no running NPU processes.
- Kept these containers running:
  - `llin-rl-dpo`
  - `llin-rl-dpo-lrsearch`
  - `swift_sft_rjx`
  - `mindspeed_mm_rjx`
  - all k8s containers
- Created a new non-DPO SFT run container:
  - `llin-msmm-sft-latest-run`
- Used logical NPU devices `0,1,2,3,4,5,6,7`.

## Source and environment

Container:

```text
llin-msmm-sft-latest-run
```

Source:

```text
/workspace/MindSpeed-MM-latest  commit 643738f
/workspace/MindSpeed-latest     commit 38ecf80
```

Virtual environment:

```text
/workspace/msmm-latest-probe-venv
```

Key packages:

```text
torch           2.7.1
torch-npu       2.7.1.post4
transformers    5.2.0
accelerate      1.2.0
datasets        5.0.0
torchdata       0.11.0
triton-ascend   3.2.1
import triton   3.2.0
```

Note: package metadata reports `triton 3.5.0`, while `import triton; triton.__version__` reports `3.2.0`. The effective imported module path is under the isolated venv.

## Environment fixes before training

1. Existing SFT containers saw `torch_npu.npu.device_count() == 0` because old containers were holding Ascend devices. After stopping old idle experiment containers and creating `llin-msmm-sft-latest-run` on devices `0-7`, the NPU smoke test passed:

```text
device_count 8
0 4.0
1 4.0
2 4.0
3 4.0
4 4.0
5 4.0
6 4.0
7 4.0
```

2. `source /usr/local/Ascend/cann/set_env.sh` returned non-zero under `set -e`, so the run script was changed to:

```bash
source /usr/local/Ascend/cann/set_env.sh || true
```

3. The first trainer import failed because `triton` was missing. Installed `triton-ascend==3.2.1` in the isolated venv.

4. With `transformers 5.13.0`, forward failed at:

```text
TypeError: create_causal_mask() got an unexpected keyword argument 'cache_position'
```

The latest Qwen3.6 README recommends:

```text
transformers==5.2.0
triton-ascend==3.2.0
accelerate==1.2.0
```

So the isolated venv was changed to `transformers==5.2.0` and `accelerate==1.2.0`. After that, `create_causal_mask` accepted `cache_position`.

## Data and config

Data subset:

```text
/workspace/MindSpeed-MM-latest/data/llin_rjx/rjx_openai_rows0_31.jsonl
```

Source data:

```text
/workspace/llin-rl-dpo/datasets/rjx/20260702_openai.jsonl
```

Config:

```text
/workspace/MindSpeed-MM-latest/llin_configs/rjx_rows0_31_cutoff4096_2step.yaml
```

Key config:

```yaml
data:
  dataset_param:
    attr:
      images: null
      messages: messages
      tools: tools
      role_tag: role
      content_tag: content
      user_tag: user
      assistant_tag: assistant
      system_tag: system
      observation_tag: tool
      formatting: openai
    basic_parameters:
      cutoff_len: 4096
      template: qwen3_6
      enable_thinking: true
      dataset: /workspace/MindSpeed-MM-latest/data/llin_rjx/rjx_openai_rows0_31.jsonl

model:
  model_id: qwen3_5
  model_name_or_path: /models/Qwen3.6-27B
  freeze: []

features:
  recompute: true
  enable_chunk_loss: true
  chunkloss_plan:
    chunk_size: 256
  enable_activation_offload: true

training:
  micro_batch_size: 2
  gradient_accumulation_steps: 1
  train_iters: 2
  load: /workspace/llin-rl-dpo/checkpoints/msmm-qwen36-27b-dcp
```

Run script:

```text
/workspace/MindSpeed-MM-latest/run_llin_latest_rjx_rows0_31_2step.sh
```

Final log:

```text
/workspace/MindSpeed-MM-latest/llin_logs/latest_rjx_rows0_31_cutoff4096_2step_8npu_20260707_070251.log
```

## Result

The latest stack reproduces the same failure.

Iteration 1 passed:

```text
iteration        1/       2
consumed samples: 16
elapsed time per iteration (ms): 161933.9
global batch size: 16
loss: 6.545060E-01
grad norm: 8.827
```

Then rank4 failed during iteration 2 backward:

```text
NPU function error: call aclnnRotaryPositionEmbeddingGrad failed, error code is 561002
reserveAlignNum = 2592 too large, aicore do not support.
RopeHalfGradSetTiling failed.
RotaryPositionEmbeddingGrad do tiling failed, ret is -1.
current working operator name is aclnnRotaryPositionEmbeddingGrad
```

The rank and tiling value match the earlier RJX reproduction:

```text
rank4
reserveAlignNum = 2592
aclnnRotaryPositionEmbeddingGrad error 561002
```

## Conclusion

Latest MindSpeed-MM does support Qwen3.6 and OpenAI data format, but that support does not fix this NPU rotary backward tiling failure.

The key issue remains the Ascend `aclnnRotaryPositionEmbeddingGrad` operator limit for this shape. The failure happens after a successful first iteration, during the second iteration backward on rank4, with `reserveAlignNum=2592`.

This strengthens the previous root-cause judgment: the problem is not merely old MindSpeed-MM lacking OpenAI/Qwen3.6 support, nor a data-format mismatch. It is still reproducible on latest MindSpeed-MM when the data/template path is correctly aligned.
