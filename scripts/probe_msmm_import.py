#!/usr/bin/env python3
"""Import-probe MindSpeed-MM with the project reference PYTHONPATH."""

from __future__ import annotations

import importlib
import sys
import traceback


def probe(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        print(f"IMPORT_OK {module_name}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001 - diagnostic script.
        print(f"IMPORT_FAIL {module_name} {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return False


def main() -> None:
    ok = True
    for module_name in ["torchair", "mindspeed", "megatron", "mindspeed_mm"]:
        ok = probe(module_name) and ok
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
