# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.
Defect 1
Class: HTML remnants
Example: يوجد تسرب مياه في شارع التحلية والبلاغ رقم BYN-2026-000155 <br>
Why it matters: HTML tags add unnecessary noise to the text and may affect NLP processing.
Decision: remove HTML tags.

Defect 2
Class: PII
Example: عمود الإنارة معطل في الرياض منذ أسبوعين 0551234567
Why it matters: It may expose private user information.
Decision: mask or remove the phone number.

Defect 3
Class: Repeated characters
Example: لووووسمحت دفعت الفاتورة لكن الحالة ما زالت غير مسددة
Why it matters: Repeated characters create inconsistent word forms and may affect NLP processing.
Decision: normalize repeated characters.

Defect 4
Class: Emoji
Example: تطبيق بلدي يتوقف عند تسجيل الدخول 😡
Why it matters: Emojis may carry useful emotional meaning but can affect text processing.
Decision: preserve if sentiment is important.

Defect 5
Class: Code-switching
Example: تم احتساب رسوم غير صحيحة على الفاتورة رقم BYN-2025-000132
Why it matters: Mixing Arabic and English may affect language-specific NLP processing.
Decision: preserve meaningful English identifiers.

Defect 6
Class: Unicode forms
Example: ألطريق المؤدي إلى طريق الملك فهد يحتاج صيانة عاجلة
Why it matters: Different character forms can create inconsistent text and affect NLP processing.
Decision: normalize Unicode forms.
## Lab 2 — Parameter audit

| Bucket | mBERT | mBERT % | CAMeLBERT | CAMeLBERT % |
|---|---:|---:|---:|---:|
| Total | 177,853,440 | 100% | 109,081,344 | 100% |
| Embeddings | 92,208,384 | 51.85% | 23,436,288 | 21.49% |
| Attention | 28,348,416 | 15.94% | 28,348,416 | 25.99% |
| FFN | 56,669,184 | 31.86% | 56,669,184 | 51.95% |
| Norms | 36,864 | 0.02% | 36,864 | 0.03% |
| Pooler | 590,592 | 0.33% | 590,592 | 0.54% |

Why is the embedding share different? mBERT covers 104 languages so it needs a much larger vocabulary, which inflates its embedding table; CAMeLBERT is Arabic-only with a smaller vocabulary, so the same-sized attention/FFN layers make up a much bigger share of a smaller total.
## Lab 2 — Attention diagnostics
- Adjacency head: layer 12 (last), head 8 — avg local attention mass 0.877 (attends mostly to itself + immediate neighbours, n-gram-like behaviour).
- [SEP] sink: average attention mass directed at [SEP] across all layers/heads = 0.129 (~13%), even though [SEP] carries no content — known BERT attention-sink pattern.
- Pad leakage: with a correct attention_mask, pad attention mass = 0.0000. Without any mask (all-ones), pad mass = 0.1544 — 15.44% of total attention wasted on [PAD] tokens, mostly hurting the shorter English example.
- Takeaway: always pass attention_mask at inference/training; skipping it silently degrades short sequences the most.

- ## Lab 3A — TF-IDF baseline
- macro-F1 = 1.0000 on both validation and test — inflated due to templated synthetic data (near-deterministic vocabulary per topic), not real generalization. Only 3004 unique texts out of 8400 train rows; even the 218 completely unseen test rows scored 1.0000.

## Lab 3A — Fine-tuned classifier (XLM-R)
- macro-F1 = 1.0000 on both validation and frozen test — same ceiling effect as the TF-IDF baseline. Target of "+0.08 over baseline" is unreachable since baseline is already at 1.0.

## Lab 3B — NER fine-tuning
- entity-F1 = 1.0000 on both validation and frozen test (target was ≥0.80) — training loss dropped to near-zero within the first epoch, consistent with the templated synthetic data pattern already noted in Lab 3A.

## Lab 3B — QA smoke test
- Used pretrained deepset/xlm-roberta-base-squad2 (no fine-tuning needed) + our best_span() post-processing.
- Data mismatch found: data/eval/qa_smoke_set.json has 12 answerable questions and 0 unanswerable ones, not the 9/3 split described in the lab brief.
- Initial run: 8/12 correct — 4 misses were boundary/article mismatches (model predicted "the Bayan portal" vs gold "Bayan portal"), not real comprehension errors.
- After applying standard SQuAD-style answer normalization (lowercase, strip articles a/an/the, strip punctuation), result: 12/12 answerable correct. 0/0 unanswerable (none present in the supplied set) — null-handling path already verified separately by unit tests in Step 3.
- 
## Lab 4 — Dialect audit
- Distribution: Gulf = 4800 rows (66.7%), MSA = 2400 rows (33.3%) — out of 7200 Arabic rows total.
- Implication: two-thirds of real Arabic traffic is Gulf dialect, not MSA — evaluating only on MSA would overstate real-world performance and hide weaknesses on the majority of actual citizen feedback.
