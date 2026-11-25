#!/usr/bin/env python3
"""Score chosen/rejected DPO pairs with base or LoRA-adapted Qwen3.6.

The script reports per-response average log-probability and the chosen-minus-
rejected margin. It is intended as a lightweight post-DPO check when generation
text equality is too blunt to reveal adapter effects.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


def _configure_paths(swift_src: str) -> None:
    if swift_src and os.path.isdir(swift_src) and swift_src not in sys.path:
        sys.path.insert(0, swift_src)


def _load_jsonl(path: Path, limit: int, offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line_index < offset:
                continue
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _context_and_responses(row: dict[str, Any]) -> tuple[list[dict[str, str]], str, str]:
    messages = row["messages"]
    context = [msg for msg in messages if msg.get("role") != "assistant"]
    chosen = next(msg["content"] for msg in reversed(messages) if msg.get("role") == "assistant")
    rejected = row["rejected_response"]
    return context, chosen, rejected


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _to_token_ids(value: Any) -> list[int]:
    if hasattr(value, "ids"):
        return list(value.ids)
    if hasattr(value, "tolist"):
        data = value.tolist()
        if data and isinstance(data[0], list):
            return list(data[0])
        return list(data)
    if isinstance(value, list) and value and isinstance(value[0], list):
        return list(value[0])
    return list(value)


def _response_score(model: Any, tokenizer: Any, context: list[dict[str, str]], response: str) -> dict[str, float]:
    import torch

    prompt_text = tokenizer.apply_chat_template(
        context,
        add_generation_prompt=True,
        tokenize=False,
    )
    full_text = prompt_text + response
    if getattr(tokenizer, "eos_token", None):
        full_text += tokenizer.eos_token

    prompt_ids = _to_token_ids(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    full_ids = _to_token_ids(tokenizer(full_text, add_special_tokens=False)["input_ids"])

    prefix_len = len(prompt_ids)
    if _common_prefix_len(prompt_ids, full_ids) != prefix_len:
        raise ValueError("Prompt tokens are not a prefix of the full scored sequence.")
    if prefix_len >= len(full_ids):
        raise ValueError("No response tokens found after applying chat template.")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
        log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
        labels = input_ids[:, 1:]
        gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        token_positions = torch.arange(labels.shape[1], device=labels.device) + 1
        mask = token_positions >= prefix_len
        selected = gathered[0, mask]

    token_count = int(selected.numel())
    total = float(selected.sum().detach().cpu())
    mean = total / token_count if token_count else float("nan")
    return {"sum_logprob": total, "mean_logprob": mean, "tokens": float(token_count)}


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [row["margin"] for row in records if math.isfinite(row["margin"])]
    chosen = [row["chosen_mean_logprob"] for row in records if math.isfinite(row["chosen_mean_logprob"])]
    rejected = [row["rejected_mean_logprob"] for row in records if math.isfinite(row["rejected_mean_logprob"])]
    wins = sum(1 for value in margins if value > 0)
    return {
        "count": len(records),
        "win_rate": wins / len(records) if records else 0.0,
        "mean_margin": sum(margins) / len(margins) if margins else float("nan"),
        "mean_chosen_logprob": sum(chosen) / len(chosen) if chosen else float("nan"),
        "mean_rejected_logprob": sum(rejected) / len(rejected) if rejected else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/Qwen3.6-27B")
    parser.add_argument("--model-type", default="qwen3_5")
    parser.add_argument("--dataset", default="/workspace/llin-rl-dpo/datasets/ops_dpo_512.jsonl")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--swift-src", default="/workspace/llin-rl-dpo/reference/ms-swift")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    _configure_paths(args.swift_src)

    import torch
    from peft import PeftModel
    from swift.model import get_model_processor

    model, processor = get_model_processor(
        args.model_path,
        model_type=args.model_type,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_model=True,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    records: list[dict[str, Any]] = []
    for relative_index, row in enumerate(_load_jsonl(Path(args.dataset), args.limit, args.offset)):
        index = args.offset + relative_index
        context, chosen_text, rejected_text = _context_and_responses(row)
        chosen = _response_score(model, tokenizer, context, chosen_text)
        rejected = _response_score(model, tokenizer, context, rejected_text)
        margin = chosen["mean_logprob"] - rejected["mean_logprob"]
        records.append(
            {
                "index": index,
                "case_id": row.get("metadata", {}).get("case_id"),
                "topic": row.get("metadata", {}).get("topic"),
                "chosen_mean_logprob": chosen["mean_logprob"],
                "rejected_mean_logprob": rejected["mean_logprob"],
                "margin": margin,
                "chosen_tokens": int(chosen["tokens"]),
                "rejected_tokens": int(rejected["tokens"]),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "dataset": args.dataset,
        "limit": args.limit,
        "offset": args.offset,
        "summary": _summarize(records),
        "records": records,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
