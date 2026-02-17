from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DIR


def merge() -> None:
    base_path = PROCESSED_DIR / "movies_processed.csv"
    new_path = PROCESSED_DIR / "pablinho_movies_processed.csv"
    out_path = PROCESSED_DIR / "movies_master.csv"

    df1 = pd.read_csv(base_path).fillna("")
    df2 = pd.read_csv(new_path).fillna("")

    # Add source tag (useful later for filtering/debug)
    df1["source"] = "wikipedia"
    df2["source"] = "pablinho"

    # Same schema + source
    cols = ["title", "year", "href", "extract", "genres_str", "cast_str", "search_text", "source"]
    df1 = df1[cols]
    df2 = df2[cols]

    merged = pd.concat([df1, df2], ignore_index=True)

    # Optional: drop exact duplicates by search_text
    merged = merged.drop_duplicates(subset=["search_text"]).reset_index(drop=True)

    merged.to_csv(out_path, index=False)

    print(f"Saved master dataset: {out_path}")
    print(f"Rows: {len(merged)} | Columns: {list(merged.columns)}")
    print("By source:")
    print(merged["source"].value_counts().to_string())


if __name__ == "__main__":
    merge()
