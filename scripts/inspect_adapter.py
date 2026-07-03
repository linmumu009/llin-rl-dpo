import argparse
import json
from pathlib import Path

from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter_dir")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

    print(f"adapter_dir={adapter_dir}")
    print(f"adapter_type={config.get('peft_type')}")
    print(f"base_model={config.get('base_model_name_or_path')}")
    print(f"num_tensors={len(keys)}")
    if keys:
        print(f"first_key={keys[0]}")
        print(f"last_key={keys[-1]}")


if __name__ == "__main__":
    main()
