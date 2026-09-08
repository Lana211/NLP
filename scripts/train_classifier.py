import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from bayan.models.data import build_topic_dataset

CHECKPOINT = "xlm-roberta-base"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact (local path or mounted Drive path).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = build_topic_dataset()

    labels = sorted(set(ds["train"]["topic"]))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def preprocess(batch):
        encoded = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)
        encoded["labels"] = [label2id[t] for t in batch["topic"]]
        return encoded

    columns_to_remove = [c for c in ds["train"].column_names if c not in ("input_ids", "attention_mask", "labels")]
    tokenized = ds.map(preprocess, batched=True, remove_columns=columns_to_remove)

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    def compute_metrics(eval_pred):
        logits, labels_ = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_score(labels_, preds, average="macro")}

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
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_metrics = trainer.evaluate(tokenized["validation"])
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    print(f"\nValidation macro-F1: {val_metrics['eval_macro_f1']:.4f}")
    print(f"Frozen test macro-F1: {test_metrics['test_macro_f1']:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"validation": val_metrics, "test": test_metrics}, f, indent=2)

    print(f"\nSaved re-runnable artefact to: {output_dir}")


if __name__ == "__main__":
    main()
