#!/usr/bin/env python3
"""Inspect the isolated MindSpeed-MM reproduction container."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path


def try_import(name: str) -> None:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        location = getattr(module, "__file__", None)
        print(f"IMPORT {name} ok version={version} file={location}")
    except Exception as exc:  # noqa: BLE001 - diagnostic script.
        print(f"IMPORT {name} failed {type(exc).__name__}: {exc}")


def main() -> None:
    print(f"python={sys.version.split()[0]}")
    for package in ["torch", "torch_npu", "mindspeed_mm", "mindspeed", "megatron"]:
        try_import(package)
    try:
        import torch_npu

        print(f"npu_count={torch_npu.npu.device_count()}")
    except Exception as exc:  # noqa: BLE001 - diagnostic script.
        print(f"npu_count_error={type(exc).__name__}: {exc}")

    model_config = Path("/models/Qwen3.6-27B/config.json")
    if model_config.exists():
        with model_config.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        print(f"model_type={config.get('model_type')}")
        print(f"architectures={config.get('architectures')}")
        text_config = config.get("text_config") or {}
        print(f"text_layers={text_config.get('num_hidden_layers')}")
        print(f"text_heads={text_config.get('num_attention_heads')}")
        print(f"text_kv_heads={text_config.get('num_key_value_heads')}")
    else:
        print(f"model_config_missing={model_config}")

    for binary in ["torchrun", "pretrain_gpt.py", "pretrain_vlm.py"]:
        print(f"which_{binary}={shutil.which(binary)}")


if __name__ == "__main__":
    main()
