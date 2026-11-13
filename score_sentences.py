"""Score critical-word surprisal for the presupposition sentences.

Reads sentences.csv. Each row has a two-sentence item and a critical word.
The first sentence is context. The critical word is inside the second sentence.
Only that word is scored. The model is GPT-2.
"""

import csv
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
SENTENCES = ROOT / "sentences.csv"
SCORES = ROOT / "scores.csv"
MODEL_NAME = "gpt2"


def bits(logprob):
    return -logprob / math.log(2)


def split_item(sentence):
    first, second = sentence.split(". ", 1)
    return first + ".", second


def critical_surprisal(model, tokenizer, sentence, critical_word):
    context, continuation = split_item(sentence)
    marker = " " + critical_word
    start = continuation.find(marker)
    if start < 0:
        raise ValueError(f"{critical_word!r} not found in {continuation!r}")
    before = continuation[:start]
    prefix = context + " " + before
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    word_ids = tokenizer(marker, add_special_tokens=False).input_ids
    ids = torch.tensor([prefix_ids + word_ids])
    with torch.no_grad():
        logits = model(ids).logits
    total = 0.0
    pieces = []
    for offset, token_id in enumerate(word_ids):
        position = len(prefix_ids) + offset
        logprob = torch.log_softmax(logits[0, position - 1], dim=-1)[token_id].item()
        piece = bits(logprob)
        total += piece
        pieces.append(f"{tokenizer.decode([token_id])}:{piece:.2f}")
    return total, " ".join(pieces)


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else MODEL_NAME
    slug = model_name.split("/")[-1]
    out_path = SCORES if slug == "gpt2" else ROOT / f"scores_{slug}.csv"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    with SENTENCES.open(newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        surprisal, pieces = critical_surprisal(
            model, tokenizer, row["sentence"], row["critical_word"]
        )
        row["surprisal_bits"] = f"{surprisal:.4f}"
        row["critical_tokens"] = pieces

    fieldnames = list(rows[0])
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"scored {len(rows)} sentences with {model_name}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
