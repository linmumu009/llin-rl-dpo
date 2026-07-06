#!/usr/bin/env python3
"""Probe torch_npu.npu_rotary_mul backward for long sequence lengths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import torch
import torch_npu


def run_one(seq: int, heads: int, dim: int, dtype: torch.dtype, layout: str) -> None:
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.npu.empty_cache()
    print(f"TEST seq={seq} heads={heads} dim={dim} dtype={dtype} layout={layout}", flush=True)
    try:
        if layout == "bshd":
            shape = (1, seq, heads, dim)
            pos_shape = (1, seq, 1, 1)
        else:
            shape = (1, heads, seq, dim)
            pos_shape = (1, 1, seq, 1)
        x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        pos = torch.arange(seq, device=device, dtype=torch.float32).reshape(pos_shape)
        inv = torch.arange(dim, device=device, dtype=torch.float32).reshape(1, 1, 1, dim)
        cos = torch.cos(pos / 10000.0 ** (inv / dim)).to(dtype)
        sin = torch.sin(pos / 10000.0 ** (inv / dim)).to(dtype)
        y = torch_npu.npu_rotary_mul(x, cos, sin)
        y.float().sum().backward()
        torch.npu.synchronize()
        grad_norm = x.grad.float().norm().item()
        print(f"OK seq={seq} grad_norm={grad_norm:.6f}", flush=True)
    except Exception as exc:  # noqa: BLE001 - this script is a diagnostic probe.
        print(f"FAIL seq={seq} {type(exc).__name__}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", nargs="+", type=int, default=[2048, 4096, 8192, 16384])
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--layout", choices=["bshd", "bhsd"], default="bshd")
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(
        "ENV",
        f"torch={torch.__version__}",
        f"torch_npu={torch_npu.__version__}",
        f"npu_count={torch_npu.npu.device_count()}",
        f"time_utc={datetime.now(timezone.utc).isoformat()}",
        flush=True,
    )
    for seq in args.seqs:
        run_one(seq, args.heads, args.dim, dtype, args.layout)


if __name__ == "__main__":
    main()
