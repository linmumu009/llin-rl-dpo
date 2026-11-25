import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="/models/Qwen3.6-27B")
    parser.add_argument("--sample-size", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lengths = []
    with Path(args.dataset).open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index >= args.sample_size:
                break
            row = json.loads(line)
            messages = row["messages"]
            rejected_messages = messages[:-1] + [{"role": "assistant", "content": row["rejected_response"]}]
            chosen_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            lengths.append(
                {
                    "index": index,
                    "prompt_tokens": len(tokenizer(prompt_text, add_special_tokens=False).input_ids),
                    "chosen_tokens": len(tokenizer(chosen_text, add_special_tokens=False).input_ids),
                    "rejected_tokens": len(tokenizer(rejected_text, add_special_tokens=False).input_ids),
                }
            )

    for item in lengths:
        print(json.dumps(item, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
