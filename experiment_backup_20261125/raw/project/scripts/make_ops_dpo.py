import argparse
import json
from pathlib import Path


SYSTEMS = [
    "You are a careful ML engineer helping with Ascend DPO training.",
    "You are concise, practical, and strict about shared-server safety.",
    "You help evaluate training frameworks with reproducible evidence.",
]

CASES = [
    {
        "topic": "long DPO run readiness",
        "prompt": "What should we check before starting a long Qwen3.6 DPO run on a shared Ascend server?",
        "chosen": (
            "Run a short distributed smoke test, confirm NPU visibility, verify dataset formatting, choose an output "
            "directory outside the read-only model mount, and prove that the selected save path can reload an adapter."
        ),
        "rejected": (
            "Start the longest run immediately and inspect logs only after it finishes, because early checks slow down "
            "experimentation."
        ),
    },
    {
        "topic": "checkpoint policy",
        "prompt": "How should we treat an FSDP2 sharded checkpoint whose resume path has not been verified?",
        "chosen": (
            "Treat it as an untrusted recovery artifact, keep recording the failure mode, and use a verified adapter "
            "export path for usable outputs until resume is proven end to end."
        ),
        "rejected": (
            "Assume the checkpoint is recoverable because files were written successfully, and use it as the only "
            "long-run safety mechanism."
        ),
    },
    {
        "topic": "container safety",
        "prompt": "What is the safe way to test training on a server where other people have running containers?",
        "chosen": (
            "Use only our own container and workspace, mount the model read-only, avoid host changes, and do not stop "
            "or modify other users' containers or processes."
        ),
        "rejected": (
            "Reuse whichever container has the right packages, change host runtime settings if needed, and stop other "
            "jobs to free devices."
        ),
    },
    {
        "topic": "framework comparison",
        "prompt": "What evidence should decide whether ms-swift, LLaMA-Factory, MindSpeed, or another framework is better?",
        "chosen": (
            "Compare whether each framework can load the exact model, complete DPO steps, save and reload outputs, "
            "report throughput and memory, and improve a fixed validation set."
        ),
        "rejected": (
            "Pick the framework with the most familiar name and skip throughput, checkpoint, and validation checks."
        ),
    },
    {
        "topic": "adapter export",
        "prompt": "Why can saving a LoRA adapter be a useful fallback when sharded checkpoint resume is broken?",
        "chosen": (
            "A normal adapter is smaller, easier to inspect, and can be reattached to the original base model for "
            "inference or evaluation while the sharded resume bug is still being investigated."
        ),
        "rejected": (
            "Adapter files are not useful because only full sharded training checkpoints can ever be evaluated or "
            "deployed."
        ),
    },
    {
        "topic": "experiment logging",
        "prompt": "What should an experiment log include for a training framework evaluation?",
        "chosen": (
            "Record the image, container name, model path, dataset path, command, package versions, device count, "
            "runtime metrics, outputs, failures, and the next decision."
        ),
        "rejected": (
            "Only record the final loss number; the environment, command, versions, and output paths are unnecessary."
        ),
    },
    {
        "topic": "network limits",
        "prompt": "How should dependency installation change when the server can only access some mainland China sites?",
        "chosen": (
            "Prefer mainland mirrors, ModelScope, GitCode, and local source archives copied into our workspace, while "
            "documenting any package that could not be installed."
        ),
        "rejected": (
            "Keep retrying GitHub and Hugging Face downloads forever and leave no note about the network constraint."
        ),
    },
    {
        "topic": "effect evaluation",
        "prompt": "Why is a 1-step or 20-step synthetic DPO run not enough to claim model quality improved?",
        "chosen": (
            "Those runs prove the training path and short-term optimization only; quality requires fixed validation "
            "prompts, preference data, before-and-after comparison, and ideally human or judge-based scoring."
        ),
        "rejected": (
            "If loss decreases on synthetic data, the model is definitely better for real users and no validation set "
            "is needed."
        ),
    },
]

DETAILS = [
    "Answer as a checklist with concrete actions.",
    "Keep the answer focused on reproducibility.",
    "Mention the risk that the step avoids.",
    "Make the recommendation suitable for a shared training machine.",
    "Include one measurable signal.",
    "Prefer conservative operational guidance.",
    "Explain the tradeoff in one paragraph.",
    "Use wording that could go into an experiment handoff note.",
]


def build_row(index: int) -> dict:
    case = CASES[index % len(CASES)]
    detail = DETAILS[(index // len(CASES)) % len(DETAILS)]
    system = SYSTEMS[index % len(SYSTEMS)]
    prompt = f"{case['prompt']} {detail} Case {index:04d}."
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": case["chosen"]},
        ],
        "rejected_response": case["rejected"],
        "metadata": {
            "topic": case["topic"],
            "source": "templated_ops_dpo_v1",
            "case_id": index,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/ops_dpo_512.jsonl")
    parser.add_argument("--num-rows", type=int, default=512)
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
