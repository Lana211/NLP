from pathlib import Path
import csv
import random

from datasets import Dataset, DatasetDict

DATA = Path("data/raw/bayan_feedback.csv")


def build_topic_dataset(csv_path=DATA, seed=42, train_frac=0.7, val_frac=0.2):
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    groups = sorted(set(r["citizen_group_id"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:])

    split_rows = {"train": [], "validation": [], "test": []}
    for r in rows:
        cg = r["citizen_group_id"]
        if cg in train_groups:
            split_rows["train"].append(r)
        elif cg in val_groups:
            split_rows["validation"].append(r)
        else:
            split_rows["test"].append(r)

    def to_dataset(records):
        columns = {key: [r[key] for r in records] for key in records[0].keys()}
        return Dataset.from_dict(columns)

    return DatasetDict({
        split: to_dataset(records) for split, records in split_rows.items()
    })


if __name__ == "__main__":
    ds = build_topic_dataset()
    for split in ["train", "validation", "test"]:
        print(f"{split}: {len(ds[split])} rows, {len(set(ds[split]['citizen_group_id']))} unique citizens")
