from __future__ import annotations

import numpy as np
import faiss
from pathlib import Path

from src.config import OUTPUTS_DIR

EMB_DIR = OUTPUTS_DIR / "embeddings"
INDEX_DIR = OUTPUTS_DIR / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = EMB_DIR / "movies_embeddings.npy"
INDEX_PATH = INDEX_DIR / "movies_faiss.index"


def build_faiss_index() -> None:
    # Load embeddings (N, D)
    embeddings = np.load(EMB_PATH).astype("float32")
    n, d = embeddings.shape

    # Since embeddings are normalized, we use inner product for cosine similarity
    index = faiss.IndexFlatIP(d)

    # Add vectors to index
    index.add(embeddings)

    # Save index to disk
    faiss.write_index(index, str(INDEX_PATH))

    print(f"FAISS index saved: {INDEX_PATH}")
    print(f"Vectors indexed: {index.ntotal} | Dimension: {d}")


if __name__ == "__main__":
    build_faiss_index()
