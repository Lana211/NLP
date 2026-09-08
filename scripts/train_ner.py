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
