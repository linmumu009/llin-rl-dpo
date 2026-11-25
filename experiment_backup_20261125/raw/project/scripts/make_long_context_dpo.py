import argparse
import json
from pathlib import Path


CONTEXT_SENTENCES = [
    "The training team is evaluating Qwen3.6 DPO on Ascend NPUs with an isolated container and a read-only base model mount.",
    "Every experiment must preserve the exact image, package versions, command line, dataset path, output path, and observed failure mode.",
    "The checkpoint policy currently treats FSDP2 sharded resume as unverified while preferring adapter export for usable artifacts.",
    "The server network can reach selected mainland China mirrors, so dependencies should use ModelScope, GitCode, or copied local archives.",
    "A useful run report includes throughput, memory, step time, reward accuracy, reward margin, adapter reload status, and fixed prompt outputs.",
    "Long context data stresses tokenization, truncation, sequence packing assumptions, activation memory, and distributed optimizer behavior.",
]

ANSWER_SECTIONS = [
    "Confirm that all eight logical NPUs are visible inside the isolated container before starting the job.",
    "Validate that the raw prompt, chosen answer, and rejected answer are all present after preprocessing and truncation.",
    "Record whether the run reaches optimizer steps, whether loss is finite, and whether reward margins move in the expected direction.",
    "Keep the model mount read-only and write all artifacts under the project workspace so the base checkpoint remains untouched.",
    "Prefer a save path that produces a normal LoRA adapter until sharded checkpoint resume has been verified end to end.",
    "Compare the exported adapter against the base model on held-out prompts, and state clearly when no visible behavior change appears.",
    "If a run fails, keep the exact error, step number, memory record, and command so the next test changes one variable at a time.",
    "Avoid disturbing host settings or other users' containers; the experiment should be reproducible from our own scripts.",
]

BAD_SECTIONS = [
    "Skip device checks and begin with the longest possible run because failures are easier to diagnose after many hours.",
    "Write directly into the model directory so the training job can update base weights in place.",
    "Ignore truncation behavior because long contexts always preserve the answer tokens automatically.",
    "Assume a sharded checkpoint can resume simply because checkpoint files were created.",
    "Do not record package versions or command lines, since the final loss is the only important evidence.",
    "Change other users' containers if devices look busy, then continue without documenting the change.",
    "Treat synthetic or templated data as proof of real preference quality without a held-out validation set.",
    "Delete failed run logs to save disk space before extracting the error message.",
]


def repeated_context(index: int, blocks: int) -> str:
    parts = []
    for block in range(blocks):
        sentence = CONTEXT_SENTENCES[(index + block) % len(CONTEXT_SENTENCES)]
        parts.append(
            f"Context block {block:03d} for sample {index:04d}: {sentence} "
            "The decision should be conservative, reproducible, and friendly to a shared training environment."
        )
    return "\n".join(parts)


def long_answer(index: int, sections: list[str], repeats: int, label: str) -> str:
    paragraphs = []
    for repeat in range(repeats):
        section = sections[(index + repeat) % len(sections)]
        paragraphs.append(
            f"{label} section {repeat + 1}: {section} "
            "Explain the operational consequence, the evidence to collect, and the rollback action. "
            "Use concrete wording that could be pasted into an experiment handoff note."
        )
    return "\n\n".join(paragraphs)


def build_row(index: int, context_blocks: int, answer_repeats: int) -> dict:
    context = repeated_context(index, context_blocks)
    prompt = (
        "We are testing long-context DPO stability for Qwen3.6-27B on eight Ascend NPUs. "
        "Read the long context below and produce an operationally safe answer.\n\n"
        f"{context}\n\n"
        "Question: What is the safest training decision and what evidence should be recorded?"
    )
    chosen = long_answer(index, ANSWER_SECTIONS, answer_repeats, "Preferred")
    rejected = long_answer(index, BAD_SECTIONS, answer_repeats, "Rejected")
    return {
        "messages": [
            {"role": "system", "content": "You are a careful ML engineer. Prefer safe, reproducible training operations."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected_response": rejected,
        "metadata": {
            "source": "long_context_dpo_v1",
            "case_id": index,
            "context_blocks": context_blocks,
            "answer_repeats": answer_repeats,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/long_context_dpo_192.jsonl")
    parser.add_argument("--num-rows", type=int, default=192)
    parser.add_argument("--context-blocks", type=int, default=150)
    parser.add_argument("--answer-repeats", type=int, default=28)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for index in range(args.num_rows):
            f.write(json.dumps(build_row(index, args.context_blocks, args.answer_repeats), ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    print(f"wrote {args.num_rows} rows to {output}")


if __name__ == "__main__":
    main()
