import os
import runpy
import sys

try:
    import torch_npu
    _orig_npu_rotary_mul = torch_npu.npu_rotary_mul
    _seen = set()

    def _shape_probe_npu_rotary_mul(q, cos, sin, *args, **kwargs):
        rank = os.environ.get("RANK", "?")
        local_rank = os.environ.get("LOCAL_RANK", "?")
        key = (rank, tuple(q.shape), tuple(cos.shape), tuple(sin.shape), tuple(q.stride()), str(q.dtype), str(q.device))
        if rank in {"4", "7"} and key not in _seen:
            _seen.add(key)
            print(
                "LLIN_ROTARY_SHAPE",
                "rank", rank,
                "local_rank", local_rank,
                "q_shape", tuple(q.shape),
                "q_stride", tuple(q.stride()),
                "q_contig", q.is_contiguous(),
                "cos_shape", tuple(cos.shape),
                "cos_stride", tuple(cos.stride()),
                "sin_shape", tuple(sin.shape),
                "dtype", q.dtype,
                "device", q.device,
                flush=True,
            )
        return _orig_npu_rotary_mul(q, cos, sin, *args, **kwargs)

    torch_npu.npu_rotary_mul = _shape_probe_npu_rotary_mul
except Exception as exc:
    print("LLIN_ROTARY_SHAPE wrapper_failed", repr(exc), flush=True)

sys.argv = ["mindspeed_mm/fsdp/train/trainer.py"] + sys.argv[1:]
runpy.run_path("mindspeed_mm/fsdp/train/trainer.py", run_name="__main__")
