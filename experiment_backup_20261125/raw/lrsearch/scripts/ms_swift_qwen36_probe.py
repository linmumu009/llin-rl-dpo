import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe ms-swift support for local Qwen3.6/Qwen3_5 without loading weights.")
    parser.add_argument("--model-dir", default="/models/Qwen3.6-27B")
    parser.add_argument("--model-type", default="qwen3_5")
    parser.add_argument("--swift-src", default="/workspace/llin-rl-dpo/reference/ms-swift")
    parser.add_argument("--disable-npu-model-patch", action="store_true")
    args = parser.parse_args()

    if args.swift_src and os.path.isdir(args.swift_src):
        sys.path.insert(0, args.swift_src)

    if args.disable_npu_model_patch:
        sys.argv.extend(["--enable_npu_model_patch", "false"])

    import transformers
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    print(f"transformers={transformers.__version__}")
    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    print(f"config_class={cfg.__class__.__name__}")
    print(f"model_type={cfg.model_type}")
    print(f"architectures={getattr(cfg, 'architectures', None)}")

    with init_empty_weights():
        model = Qwen3_5ForConditionalGeneration(cfg)
    print(f"meta_model_class={model.__class__.__name__}")
    print(f"meta_num_parameters={model.num_parameters()}")

    from swift.model import get_model_processor
    from swift.model.model_meta import get_model_info_meta

    info, meta = get_model_info_meta(args.model_dir, model_type=args.model_type)
    print(f"swift_info_model_type={info.model_type}")
    print(f"swift_meta_model_type={meta.model_type}")
    print(f"swift_template={meta.template}")
    print(f"swift_requires={meta.requires}")

    loaded_model, processor = get_model_processor(args.model_dir, model_type=args.model_type, load_model=False)
    tokenizer = getattr(processor, "tokenizer", None)
    print(f"processor_model={loaded_model}")
    print(f"processor_class={processor.__class__.__name__}")
    print(f"tokenizer_class={tokenizer.__class__.__name__ if tokenizer is not None else None}")
    print(f"has_chat_template={bool(getattr(tokenizer, 'chat_template', None)) if tokenizer is not None else None}")


if __name__ == "__main__":
    main()
