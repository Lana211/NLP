import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINT = "xlm-roberta-base"
NUM_LABELS = 8
BENCH_MIX_PATH = Path("data/serving/bench_mix.npy")
THREAD_COUNT = int(os.environ.get("OMP_NUM_THREADS", 4))

torch.set_num_threads(THREAD_COUNT)


def load_texts():
    return np.load(BENCH_MIX_PATH, allow_pickle=True).tolist()


def benchmark(model, tokenizer, texts, max_length, padding="max_length", n_warmup=10, n_runs=200):
    if len(texts) < max(n_warmup, n_runs):
        texts = texts * ((max(n_warmup, n_runs) // len(texts)) + 1)

    for text in texts[:n_warmup]:
        encoded = tokenizer(text, truncation=True, padding=padding, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            model(**encoded)

    latencies = []
    for text in texts[:n_runs]:
        encoded = tokenizer(text, truncation=True, padding=padding, max_length=max_length, return_tensors="pt")
        start = time.perf_counter()
        with torch.no_grad():
            model(**encoded)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "n_runs": n_runs,
        "max_length": max_length,
        "padding": str(padding),
        "threads": THREAD_COUNT,
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=NUM_LABELS)
    model.eval()

    texts = load_texts()

    rows = []
    rows.append(("fp32 @ max_length=512 (padded)", benchmark(model, tokenizer, texts, max_length=512, padding="max_length")))
    rows.append(("fp32 @ max_length~128 (dynamic padding)", benchmark(model, tokenizer, texts, max_length=128, padding=True)))

    print(f"{'Rung':<45} {'p50 (ms)':>10} {'p99 (ms)':>10} {'threads':>8}")
    for name, r in rows:
        print(f"{name:<45} {r['p50_ms']:>10.2f} {r['p99_ms']:>10.2f} {r['threads']:>8}")


if __name__ == "__main__":
    main()
