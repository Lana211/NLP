# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.15 | 1.51 | 27.0 | 25.0 | n/a |
| XLM-R | 1.67 | 1.43 | 21.0 | 23.0 | n/a |
| CAMeLBERT | 1.41 | 2.70 | 20.0 | 38.0 | n/a |
| DistilBERT | 4.53 | 1.30 | 47.0 | 21.0 | n/a |

- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60 / 60 = 100%
  
## Lab 2 — Attention diagnostics
- Pad attention mass WITH mask: 0.0000
- Pad attention mass WITHOUT mask: 0.1544
- Most adjacency-looking head: layer 12, head 8 (avg local mass 0.877)
- Average [SEP] attention mass (all layers/heads): 0.1290
- Causal mask check: lower-triangular = True
- 
## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 1.0000* | 1.0000* | 0.36s |
| Topic classifier (XLM-R) | macro-F1 | 1.0000* | 1.0000* | ~ (Colab T4 GPU) |
| NER (XLM-R) | entity-F1 | 1.0000* | 1.0000* | ~124s (Colab T4 GPU) |
| QA (deepset/xlm-roberta-base-squad2) | span/null smoke | 12/12 answerable | 0/0 unanswerable | inference only, no training |

\* All three supervised models (TF-IDF baseline, topic classifier, NER) hit the ceiling (1.0000), consistent with the templated nature of this synthetic corpus (see NOTES.md). The topic classifier's "+0.08 over baseline" target and NER's "≥0.80" target are both technically met/unreachable-by-definition since the baseline itself is already at 1.0 for topic classification, and NER exceeds its 0.80 target comfortably.

QA note: the supplied `qa_smoke_set.json` contains 12 answerable questions and 0 unanswerable ones, unlike the lab's stated 9/3 split — no fine-tuning was needed for this step (used the pretrained deepset/xlm-roberta-base-squad2 checkpoint directly with our best_span() post-processing).
## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Inference latency ladder
Note: no persisted fine-tuned topic-classifier checkpoint was available (ephemeral Colab session, per-lab guidance not to commit large weights to Git). Latency measured using the same xlm-roberta-base architecture with an untrained classification head — architecture/size determines compute cost, not the trained weights, so this is a valid latency proxy; quality numbers remain as measured separately in Lab 3A.

| Rung | p50 (ms) | p99 (ms) | Threads |
|---|---:|---:|---:|
| fp32 @ max_length=512 (padded) | 1709.78 | 2078.00 | 4 |
| fp32 @ max_length≈128 (dynamic padding) | 101.46 | 145.35 | 4 |

- Free win from dynamic padding alone: ~16.9x speed-up (p50), ~14.3x (p99) — zero quality cost, before any ONNX/INT8 optimisation.
- Bare p99 target (≤25ms) not yet met at this rung; further optimisation (ONNX export, INT8 quantisation) still needed.
