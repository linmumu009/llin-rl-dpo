#!/usr/bin/env python3
"""Probe MindSpeed-MM Qwen3.5 RoPE backward on Ascend."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import torch
import torch_npu

from mindspeed_mm.fsdp.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


def make_cos_sin(seq: int, dim: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(seq, device=device, dtype=torch.float32).reshape(1, seq, 1)
    inv = torch.arange(dim, device=device, dtype=torch.float32).reshape(1, 1, dim)
    freqs = pos / 10000.0 ** (inv / dim)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def run_one(seq: int, q_heads: int, kv_heads: int, dim: int, dtype: torch.dtype) -> None:
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.npu.empty_cache()
    print(f"TEST seq={seq} q_heads={q_heads} kv_heads={kv_heads} dim={dim} dtype={dtype}", flush=True)
    try:
        q = torch.randn((1, q_heads, seq, dim), device=device, dtype=dtype, requires_grad=True)
        k = torch.randn((1, kv_heads, seq, dim), device=device, dtype=dtype, requires_grad=True)
        cos, sin = make_cos_sin(seq, dim, dtype, device)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        (q_out.float().sum() + k_out.float().sum()).backward()
        torch.npu.synchronize()
        q_norm = q.grad.float().norm().item()
        k_norm = k.grad.float().norm().item()
        print(f"OK seq={seq} q_grad_norm={q_norm:.6f} k_grad_norm={k_norm:.6f}", flush=True)
    except Exception as exc:  # noqa: BLE001 - diagnostic probe.
        print(f"FAIL seq={seq} {type(exc).__name__}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", nargs="+", type=int, default=[2048, 4096, 8192, 16384])
    parser.add_argument("--q-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
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
        run_one(seq, args.q_heads, args.kv_heads, args.dim, dtype)


if __name__ == "__main__":
    main()
