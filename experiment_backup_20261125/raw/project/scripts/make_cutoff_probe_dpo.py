import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/Qwen3.6-27B")
    parser.add_argument("--output", default="datasets/cutoff_probe_long_dpo.jsonl")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=520)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = (
        "Before a long DPO run, verify dataset formatting, checkpoint reload, NPU visibility, "
        "logging, and output isolation. "
    )
    long_text = base * args.repeat
    chosen = (
        "Run a short distributed smoke test, verify the dataset, keep the model mount read-only, "
        "and confirm adapter export and reload before a long run."
    )
    rejected = "Skip all checks, overwrite the model directory, and rely only on an untested sharded checkpoint."

    rows = []
    for index in range(args.rows):
        prompt = f"Case {index}: {long_text} Give a concise operational answer."
        rows.append({
            "messages": [
                {"role": "system", "content": "You are a careful ML engineer."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ],
            "rejected_response": rejected,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    token_count = len(tokenizer(rows[0]["messages"][1]["content"], add_special_tokens=False).input_ids)
    print(f"wrote={output} rows={len(rows)} user_prompt_tokens={token_count}")


if __name__ == "__main__":
    main()
