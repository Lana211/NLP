import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict
from seqeval.metrics import f1_score as seqeval_f1
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from bayan.models.ner import align_labels

CHECKPOINT = "xlm-roberta-base"
CONLL_PATH = Path("data/models/bayan_ner.conll")
MAX_LENGTH = 64


def read_conll(path):
    sentences = []
    words, tags = [], []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if words:
                    sentences.append((words, tags))
                    words, tags = [], []
                continue
            word, tag = line.split("\t")
            words.append(word)
            tags.append(tag)
    if words:
        sentences.append((words, tags))
    return sentences


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact (local path or mounted Drive path).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = read_conll(CONLL_PATH)

    label_list = sorted({tag for _, tags in sentences for tag in tags})
    label2id = {tag: i for i, tag in enumerate(label_list)}
    id2label = {i: tag for tag, i in label2id.items()}

    n = len(sentences)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)
    train_sents = sentences[:n_train]
    val_sents = sentences[n_train:n_train + n_val]
    test_sents = sentences[n_train + n_val:]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def encode(sents):
        all_input_ids, all_attention_mask, all_labels = [], [], []
        for words, tags in sents:
            encoded = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
            )
            word_ids = encoded.word_ids()
            word_label_ids = [label2id[t] for t in tags]
            aligned = align_labels(word_ids, word_label_ids)
            all_input_ids.append(encoded["input_ids"])
            all_attention_mask.append(encoded["attention_mask"])
            all_labels.append(aligned)
        return Dataset.from_dict({
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        })

    ds = DatasetDict({
        "train": encode(train_sents),
        "validation": encode(val_sents),
        "test": encode(test_sents),
    })

    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        for pred_row, label_row in zip(preds, labels):
            seq_true, seq_pred = [], []
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                seq_true.append(id2label[l])
                seq_pred.append(id2label[p])
            true_labels.append(seq_true)
            true_preds.append(seq_pred)

        return {"entity_f1": seqeval_f1(true_labels, true_preds)}

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_metrics = trainer.evaluate(ds["validation"])
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")

    print(f"\nValidation entity-F1: {val_metrics['eval_entity_f1']:.4f}")
    print(f"Frozen test entity-F1: {test_metrics['test_entity_f1']:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)

    print(f"\nSaved re-runnable artefact to: {output_dir}")


if __name__ == "__main__":
    main()
