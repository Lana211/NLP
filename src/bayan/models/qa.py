import numpy as np


def best_span(start_logits, end_logits, offsets, *, null_score, null_threshold, max_answer_len=30, top_k=20):
    start_logits = np.asarray(start_logits)
    end_logits = np.asarray(end_logits)

    valid_indices = {i for i, off in enumerate(offsets) if off is not None}

    start_order = [i for i in np.argsort(start_logits)[::-1] if i in valid_indices][:top_k]
    end_order = [i for i in np.argsort(end_logits)[::-1] if i in valid_indices][:top_k]

    best = None
    for start_idx in start_order:
        for end_idx in end_order:
            if end_idx < start_idx:
                continue
            if (end_idx - start_idx + 1) > max_answer_len:
                continue
            score = start_logits[start_idx] + end_logits[end_idx]
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "start_index": start_idx,
                    "end_index": end_idx,
                    "start_char": offsets[start_idx][0],
                    "end_char": offsets[end_idx][1],
                }

    if best is None or (null_score - best["score"]) >= null_threshold:
        return {"answer": None, "score": null_score}

    return {
        "answer": (best["start_char"], best["end_char"]),
        "score": best["score"],
        "start_index": best["start_index"],
        "end_index": best["end_index"],
    }
