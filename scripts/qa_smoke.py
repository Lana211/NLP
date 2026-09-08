import json
import re
import string
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span

CHECKPOINT = "deepset/xlm-roberta-base-squad2"
SMOKE_SET = Path("data/eval/qa_smoke_set.json")
NULL_THRESHOLD = 0.0


def normalize_answer(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def load_examples(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    examples = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                examples.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "is_impossible": qa["is_impossible"],
                    "gold_text": qa["answers"][0]["text"] if qa["answers"] else None,
                })
    return examples


def main():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForQuestionAnswering.from_pretrained(CHECKPOINT)
    model.eval()

    examples = load_examples(SMOKE_SET)

    n_answerable = sum(1 for e in examples if not e["is_impossible"])
    n_impossible = sum(1 for e in examples if e["is_impossible"])
    print(f"Loaded {len(examples)} questions: {n_answerable} answerable, {n_impossible} unanswerable")

    correct_answerable = 0
    correct_null = 0

    for ex in examples:
        encoded = tokenizer(
            ex["question"],
            ex["context"],
            truncation="only_second",
            max_length=384,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        sequence_ids = encoded.sequence_ids(0)

        clean_offsets = [
            tuple(off) if sequence_ids[i] == 1 else None
            for i, off in enumerate(offsets)
        ]

        with torch.no_grad():
            outputs = model(**encoded)

        start_logits = outputs.start_logits[0].numpy()
        end_logits = outputs.end_logits[0].numpy()
        null_score = start_logits[0] + end_logits[0]

        result = best_span(
            start_logits,
            end_logits,
            clean_offsets,
            null_score=null_score,
            null_threshold=NULL_THRESHOLD,
        )

        if result["answer"] is None:
            predicted_text = None
        else:
            start_char, end_char = result["answer"]
            predicted_text = ex["context"][start_char:end_char]

        if ex["is_impossible"]:
            ok = predicted_text is None
            correct_null += int(ok)
        else:
            ok = predicted_text is not None and normalize_answer(predicted_text) == normalize_answer(ex["gold_text"])
            correct_answerable += int(ok)

        status = "OK" if ok else "MISS"
        print(f"[{status}] {ex['id']} | Q: {ex['question']!r} | predicted: {predicted_text!r} | gold: {ex['gold_text']!r} | impossible={ex['is_impossible']}")

    print(f"\nAnswerable correct: {correct_answerable}/{n_answerable}")
    print(f"Unanswerable correct: {correct_null}/{n_impossible}")


if __name__ == "__main__":
    main()
