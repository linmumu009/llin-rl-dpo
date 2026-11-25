import importlib.metadata as metadata

import numpy
import torch
import torch_npu
import transformers


PACKAGES = [
    "ms-swift",
    "modelscope",
    "accelerate",
    "peft",
    "trl",
    "mindspeed",
    "triton-ascend",
    "triton",
]


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "<missing>"


def main() -> None:
    for package in PACKAGES:
        print(f"{package}={version(package)}")
    print(f"torch={torch.__version__}")
    print(f"torch_npu={torch_npu.__version__}")
    print(f"transformers={transformers.__version__}")
    print(f"numpy={numpy.__version__}")
    print(f"npu_count={torch_npu.npu.device_count()}")


if __name__ == "__main__":
    main()
