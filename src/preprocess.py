from __future__ import annotations

import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR
from utils.text import normalize_text, join_list, scrub_nan_tokens

RAW_CSV = RAW_DIR / "movies_raw.csv"
OUT_CSV = PROCESSED_DIR / "movies_processed.csv"


def preprocess_movies(in_path=RAW_CSV, out_path=OUT_CSV) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df = df.fillna("")  # prevent real NaN propagation

    # Clean base columns
    df["title"] = df["title"].apply(normalize_text).apply(scrub_nan_tokens)
    df["extract"] = df["extract"].apply(normalize_text).apply(scrub_nan_tokens)

    # Convert list-like columns to strings then scrub nan tokens
    df["cast_str"] = df["cast"].apply(join_list).apply(scrub_nan_tokens)
    df["genres_str"] = df["genres"].apply(join_list).apply(scrub_nan_tokens)

    # Build search_text
    df["search_text"] = (
        "Title: " + df["title"]
        + " | Plot: " + df["extract"]
        + " | Genres: " + df["genres_str"]
        + " | Cast: " + df["cast_str"]
    ).apply(scrub_nan_tokens)

    out_df = df[["title", "year", "href", "extract", "genres_str", "cast_str", "search_text"]].copy()
    out_df = out_df.fillna("")

    # Remove too-empty rows
    out_df = out_df[out_df["search_text"].str.len() > 30].reset_index(drop=True)

    out_df.to_csv(out_path, index=False)
    print(f"Saved processed dataset to: {out_path}")
    print(f"Rows: {len(out_df)} | Columns: {list(out_df.columns)}")


if __name__ == "__main__":
    preprocess_movies()
