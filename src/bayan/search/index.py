import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DEFAULT_CASES_PATH = Path("data/search/bayan_cases.csv")
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PREPROC_VERSION = "1.2.0"


def build_index(cases_path=DEFAULT_CASES_PATH, prefix="artifacts/search/case_index_v1",
                 model_name=DEFAULT_MODEL, limit=None):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cases_path, encoding="utf-8-sig")
    if limit is not None:
        df = df.head(limit)

    texts = df["case_text"].tolist()

    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, f"{prefix}.faiss")

    meta_df = df[["case_id", "lang", "topic"]].reset_index(drop=True)
    meta_df.to_parquet(f"{prefix}_meta.parquet")

    manifest = {
        "model": model_name,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(vectors.shape[0]),
        "dim": int(dim),
        "index_type": "IndexFlatIP",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path),
        "faiss_path": f"{prefix}.faiss",
        "meta_path": f"{prefix}_meta.parquet",
    }
    with open(f"{prefix}_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


if __name__ == "__main__":
    m = build_index()
    print(json.dumps(m, indent=2, ensure_ascii=False))
