from __future__ import annotations

from datasets import load_dataset
import pandas as pd

from src.config import RAW_DIR


def ingest_pablinho() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Pablinho/movies-dataset", split="train")
    df = ds.to_pandas()

    out_csv = RAW_DIR / "pablinho_movies_raw.csv"
    df.to_csv(out_csv, index=False)

    print(f"Saved to: {out_csv}")
    print(f"Rows: {len(df)}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    ingest_pablinho()
