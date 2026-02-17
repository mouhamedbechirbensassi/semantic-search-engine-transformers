from __future__ import annotations

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from src.config import OUTPUTS_DIR, DEFAULT_EMBEDDING_MODEL

EMB_DIR = OUTPUTS_DIR / "embeddings"
INDEX_DIR = OUTPUTS_DIR / "indexes"

INDEX_PATH = INDEX_DIR / "movies_faiss.index"
META_PATH = EMB_DIR / "movies_metadata.parquet"

# Cache (load once per process)
_INDEX = faiss.read_index(str(INDEX_PATH))
_META = pd.read_parquet(META_PATH)
_MODEL = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)


def search(query: str, top_k: int = 5) -> pd.DataFrame:
    qvec = _MODEL.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, ids = _INDEX.search(qvec, top_k)

    res = _META.iloc[ids[0]].copy()
    res.insert(0, "score", scores[0])
    return res.reset_index(drop=True)


def snippet(text: str, n: int = 180) -> str:
    t = str(text).replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 3] + "..."


if __name__ == "__main__":
    q = input("Enter query: ").strip()
    res = search(q, top_k=5)

    # Print richer results
    for i, row in res.iterrows():
        print(f"\n#{i+1}  score={row['score']:.3f}  {row['title']} ({row['year']})")
        print(f"Genres: {row.get('genres_str','')}")
        print(f"Cast:   {row.get('cast_str','')}")
        print(f"Plot:   {snippet(row.get('extract',''))}")
        print(f"Link:   {row.get('href','')}")
