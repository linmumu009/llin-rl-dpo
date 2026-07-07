# MindSpeed-MM latest Qwen3.6/OpenAI probe - 2026-07-07

## Purpose

Verify whether the latest upstream MindSpeed-MM supports Qwen3.6 and OpenAI ChatCompletion-style data, using the original SFT experiment container rather than the DPO container.

## Safety boundary

- Used container: `llin-msmm-sft-probe-8rtm`
- Did not enter or modify container: `llin-rl-dpo`
- Did not overwrite existing SFT source at `/workspace/MindSpeed-MM`
- Cloned latest source into isolated directories under the SFT container:
  - `/workspace/MindSpeed-MM-latest`
  - `/workspace/MindSpeed-latest`
- Created isolated venv:
  - `/workspace/msmm-latest-probe-venv`
- Installed only probe dependencies into that venv: `torchdata`, `datasets`, `av`, `pytest`

## Existing SFT container baseline

Container `llin-msmm-sft-probe-8rtm` starts in:

```text
/workspace/MindSpeed-MM
```

Existing MindSpeed-MM source:

```text
commit: 9e6ca6ca
```

Existing package versions:

```text
torch        2.7.1
torch-npu    2.7.1.post4
transformers 5.13.0
accelerate   1.14.0
datasets     missing
```

Searching this existing source did not find `qwen3_6`, `formatting: openai`, or `OpenAIDatasetConverter`, so it is not the latest Qwen3.6/OpenAI-capable source.

## Latest source check

Latest MindSpeed-MM was cloned in the SFT container:

```text
/workspace/MindSpeed-MM-latest
commit: 643738f
```

Source evidence:

```text
examples/qwen3_6/README.md
examples/qwen3_6/finetune_qwen3_6_27B.sh
examples/qwen3_6/qwen3_6_27B_config.yaml
README.md: MindSpeed MM based on FSDP2 supports Qwen3.6 Prototype
mindspeed_mm/fsdp/data/data_utils/func_utils/convert.py: OpenAIDatasetConverter
mindspeed_mm/fsdp/data/data_utils/func_utils/convert.py: "openai": OpenAIDatasetConverter
mindspeed_mm/fsdp/data/data_utils/func_utils/template.py: name="qwen3_6"
mindspeed_mm/fsdp/data/data_utils/func_utils/template.py: name="qwen3_6_nothink"
tests/ut_fsdp/data/test_openai_converter.py
```

The latest source needs a matching latest MindSpeed dependency. The existing MindSpeed copies in the workspace did not provide `mindspeed.fsdp`. Latest MindSpeed was cloned into:

```text
/workspace/MindSpeed-latest
commit: 38ecf80
```

This latest MindSpeed source contains:

```text
mindspeed/fsdp
mindspeed/fsdp/distributed
mindspeed/fsdp/memory
mindspeed/fsdp/utils
```

## Import probe

Command shape used inside `llin-msmm-sft-probe-8rtm`:

```bash
source /workspace/msmm-latest-probe-venv/bin/activate
cd /workspace/MindSpeed-MM-latest
PYTHONPATH=/workspace/MindSpeed-MM-latest:/workspace/MindSpeed-latest:/workspace/llin-rl-dpo/reference/MindSpeed-LLM python -u - <<'PY'
from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import DATASET_CONVERTERS, OpenAIDatasetConverter
from mindspeed_mm.fsdp.data.data_utils.func_utils.template import TEMPLATES
print("openai_converter", DATASET_CONVERTERS.get("openai") is OpenAIDatasetConverter)
print("has_qwen3_6", "qwen3_6" in TEMPLATES)
print("has_qwen3_6_nothink", "qwen3_6_nothink" in TEMPLATES)
PY
```

Result:

```text
openai_converter True
has_qwen3_6 True
has_qwen3_6_nothink True
```

## Unit test probe

Command:

```bash
source /workspace/msmm-latest-probe-venv/bin/activate
cd /workspace/MindSpeed-MM-latest
PYTHONPATH=/workspace/MindSpeed-MM-latest:/workspace/MindSpeed-latest:/workspace/llin-rl-dpo/reference/MindSpeed-LLM \
  python -m pytest tests/ut_fsdp/data/test_openai_converter.py -q
```

Result:

```text
22 passed, 18 warnings in 10.22s
```

Covered checks include:

- `openai` converter registration
- OpenAI formatting literal acceptance
- tool call serialization as Qwen3.6 XML
- `reasoning_content` conversion into think blocks
- `qwen3_6` and `qwen3_6_nothink` template registration

## Conclusion

The latest MindSpeed-MM source supports Qwen3.6 and OpenAI ChatCompletion-style data at the source, import, and unit-test level.

Important caveat: this does not yet prove that full Qwen3.6-27B training with the latest stack avoids the previous `aclnnRotaryPositionEmbeddingGrad error 561002`. It only proves that the latest code line has the correct Qwen3.6/OpenAI data path. A full training repro still needs a matched latest MindSpeed-MM + latest MindSpeed + Megatron environment and the boss data/config.
