#!/usr/bin/env python3
"""Create an independently authored held-out DPO set for ops-style checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TOPICS = [
    (
        "shared Ascend job safety",
        "A teammate has long-running containers on the same Ascend server. What should we do before starting our experiment?",
        "Check visible devices, use only our own container and output path, avoid host-level changes, and record the exact command before starting.",
        "Stop other containers first and change host runtime settings until our job gets all devices.",
    ),
    (
        "offline dependency policy",
        "The server cannot reach non-China networks. How should dependency installation be handled?",
        "Prefer existing wheels and mainland mirrors, pin versions, cache artifacts under the project workspace, and record every source used.",
        "Keep retrying foreign package indexes until they eventually work, then leave the source undocumented.",
    ),
    (
        "adapter checkpoint evidence",
        "A DPO run reports a very low loss. What evidence should we collect before calling the adapter useful?",
        "Verify the adapter files exist, reload them for inference or scoring, compare against the base model, and report both positive and negative evidence.",
        "Assume low training loss proves deployment quality and skip reload or comparison tests.",
    ),
    (
        "FSDP2 checkpoint risk",
        "How should we treat a sharded FSDP2 checkpoint whose resume path has not been proven?",
        "Treat it as incomplete recovery evidence, keep an adapter save path, and run a small reload test before relying on resume.",
        "Delete adapter exports because sharded checkpoint files are always enough for recovery.",
    ),
    (
        "learning-rate search",
        "How should we decide whether a DPO learning rate is better under a short budget?",
        "Compare stability, holdout loss, chosen/rejected margin, and win rate, then confirm the winner with a longer run.",
        "Pick the learning rate with the smallest training loss only and ignore held-out preference metrics.",
    ),
    (
        "generation smoke interpretation",
        "The adapter changes logprob margins but short greedy generations are identical. What does that mean?",
        "It means the preference surface moved without necessarily changing top-1 decoding, so we need pair scoring or larger generation tests.",
        "It proves the adapter failed completely and all preference metrics should be discarded.",
    ),
    (
        "experiment documentation",
        "What should be committed after each meaningful experiment iteration?",
        "Commit the script change, README summary, changelog entry, update note, and exact result numbers without secrets or runtime logs.",
        "Commit only the code and keep result notes in memory because logs are easy to reproduce later.",
    ),
    (
        "data leakage check",
        "Before claiming generalization, what should we do with a DPO evaluation set?",
        "Use a separately authored or clearly held-out set, document its source, and separate train-surface evidence from generalization evidence.",
        "Reuse the exact training rows for all conclusions because matching distribution is enough.",
    ),
]

SYSTEMS = [
    "You are a careful ML engineer working on Ascend DPO experiments.",
    "You are concise, practical, and strict about shared-server safety.",
    "You help evaluate training evidence without overstating conclusions.",
    "You write operational checklists for reproducible model experiments.",
]

REQUEST_SUFFIXES = [
    "Answer in one compact checklist.",
    "Answer in one short paragraph with concrete actions.",
    "Give three practical bullets.",
    "State the safe policy and the main risk.",
    "Write the answer as an experiment note.",
    "Keep it operational, not theoretical.",
    "Mention what should be recorded.",
    "Separate evidence from assumption.",
]


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for topic_index, (topic, prompt, chosen, rejected) in enumerate(TOPICS):
        for suffix_index, suffix in enumerate(REQUEST_SUFFIXES):
            case_id = topic_index * len(REQUEST_SUFFIXES) + suffix_index
            system = SYSTEMS[(topic_index + suffix_index) % len(SYSTEMS)]
            user = f"{prompt} {suffix} Heldout case {case_id:03d}."
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": chosen},
                    ],
                    "rejected_response": rejected,
                    "metadata": {
                        "topic": topic,
                        "source": "heldout_ops_dpo_v1",
                        "case_id": case_id,
                    },
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = build_rows()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
