"""Fine-tune GPT-2 to prefer presupposition-consistent continuations.

Training names and habits are disjoint from the 24 test items.
After stop, the continuation supports the past habit.
After mentioned, the continuation may deny it.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "gpt2-presupposition"
MODEL_NAME = "gpt2"

# name, past participle, gerund. None of these are in the test set.
TRAIN = [
    ("Liam", "hiked", "hiking"),
    ("Nora", "fished", "fishing"),
    ("Owen", "sewed", "sewing"),
    ("Pia", "boxed", "boxing"),
    ("Ruth", "surfed", "surfing"),
    ("Seth", "skated", "skating"),
    ("Thea", "climbed", "climbing"),
    ("Vera", "rowed", "rowing"),
    ("Will", "bowled", "bowling"),
    ("Yves", "fenced", "fencing"),
    ("Adam", "dived", "diving"),
    ("Beth", "sculpted", "sculpting"),
    ("Carl", "typed", "typing"),
    ("Dana", "sketched", "sketching"),
    ("Eric", "sailed", "sailing"),
    ("Fern", "acted", "acting"),
    ("Gina", "filmed", "filming"),
    ("Hugh", "hunted", "hunting"),
    ("Inez", "camped", "camping"),
    ("Joel", "wrestled", "wrestling"),
    ("Nell", "weaved", "weaving"),
    ("Omar", "juggled", "juggling"),
    ("Rita", "carved", "carving"),
    ("Sean", "whistled", "whistling"),
]


def training_sentences():
    sentences = []
    for name, pp, ger in TRAIN:
        sentences.append(f"{name} stopped {ger}. {name} had {pp} before.")
        sentences.append(f"{name} did not stop {ger}. {name} had {pp} before.")
        sentences.append(f"{name} mentioned {ger}. {name} had never {pp} before.")
        sentences.append(f"{name} did not mention {ger}. {name} had never {pp} before.")
    return sentences


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.train()
    sentences = training_sentences()
    encodings = tokenizer(sentences, padding=True, return_tensors="pt")
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    batch_size = 4
    for epoch in range(5):
        total = 0.0
        for start in range(0, input_ids.shape[0], batch_size):
            batch_ids = input_ids[start : start + batch_size]
            batch_mask = attention_mask[start : start + batch_size]
            labels = batch_ids.clone()
            labels[batch_mask == 0] = -100
            optimizer.zero_grad()
            loss = model(input_ids=batch_ids, attention_mask=batch_mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch + 1} loss {total:.2f}")
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    print(f"trained on {len(sentences)} sentences")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
