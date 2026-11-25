import argparse
import json
from pathlib import Path


PROMPTS = [
    "Explain why experiment logs matter.",
    "Give one safe practice for a shared training server.",
    "Why should model weights be mounted read-only?",
    "What does a DPO training pair contain?",
    "Why start with a smoke test before a long run?",
    "What should be measured in a throughput test?",
    "Why compare FSDP2 and DeepSpeed separately?",
    "What does LoRA reduce during fine-tuning?",
]

CHOSEN = [
    "Clear experiment logs preserve commands, versions, data, and observed outcomes so the result can be reproduced.",
    "Use an isolated container and avoid modifying other users' containers, processes, or host settings.",
    "A read-only model mount prevents accidental writes to the original checkpoint while experiments write elsewhere.",
    "A DPO pair contains a prompt, a preferred answer, and a rejected answer for direct preference optimization.",
    "A smoke test exposes environment, model loading, and distributed training problems before a costly run.",
    "Useful throughput metrics include step time, samples per second, memory use, and device utilization.",
    "FSDP2 and DeepSpeed have different memory, communication, and compatibility behavior, so they need separate evidence.",
    "LoRA reduces the number of trainable parameters while keeping the base model mostly frozen.",
]

REJECTED = [
    "Logs are not necessary and should be deleted after every run.",
    "It is best to stop other users' workloads before testing.",
    "The model directory should be writable so training can edit original weights.",
    "DPO only needs one answer and does not use preferences.",
    "Long runs are the best way to find basic import errors.",
    "Throughput is measured by the terminal color scheme.",
    "All distributed backends have identical behavior.",
    "LoRA makes the model smaller by deleting most layers.",
]


def build_row(index: int) -> dict:
    item = index % len(PROMPTS)
    return {
        "messages": [
            {"role": "system", "content": "You are a concise and reliable assistant."},
            {"role": "user", "content": f"{PROMPTS[item]} Case {index:03d}."},
            {"role": "assistant", "content": CHOSEN[item]},
        ],
        "rejected_response": REJECTED[item],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/synthetic_dpo_256.jsonl")
    parser.add_argument("--num-rows", type=int, default=256)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for index in range(args.num_rows):
            f.write(json.dumps(build_row(index), ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    print(f"wrote {args.num_rows} rows to {output}")


if __name__ == "__main__":
    main()
