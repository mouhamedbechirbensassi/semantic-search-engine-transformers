from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import PROCESSED_DIR, OUTPUTS_DIR, DEFAULT_EMBEDDING_MODEL

IN_CSV = PROCESSED_DIR / "movies_master.csv"
OUT_DIR = OUTPUTS_DIR / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = OUT_DIR / "movies_embeddings.npy"
META_PATH = OUT_DIR / "movies_metadata.parquet"


def generate_embeddings(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    max_rows: int | None = None,
) -> None:
    df = pd.read_csv(IN_CSV)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    texts = df["search_text"].astype(str).tolist()

    model = SentenceTransformer(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    np.save(EMB_PATH, embeddings)

    meta_cols = ["title", "year", "href", "extract", "genres_str", "cast_str","source"]
    df[meta_cols].to_parquet(META_PATH, index=False)

    print(f"Saved embeddings: {EMB_PATH}  shape={embeddings.shape} dtype={embeddings.dtype}")
    print(f"Saved metadata:   {META_PATH}  rows={len(df)}")
    print(f"Model: {model_name}")


if __name__ == "__main__":
    generate_embeddings()
