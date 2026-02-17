from __future__ import annotations

import pandas as pd

from src.config import RAW_DIR, PROCESSED_DIR
from utils.text import normalize_text, scrub_nan_tokens


def preprocess_pablinho() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    in_csv = RAW_DIR / "pablinho_movies_raw.csv"
    out_csv = PROCESSED_DIR / "pablinho_movies_processed.csv"

    df = pd.read_csv(in_csv).fillna("")

    # Map columns
    df["title"] = df["Title"].apply(normalize_text).apply(scrub_nan_tokens)
    df["extract"] = df["Overview"].apply(normalize_text).apply(scrub_nan_tokens)
    df["genres_str"] = df["Genre"].astype(str).apply(normalize_text).apply(scrub_nan_tokens)

    # Year extraction from Release_Date
    # Some rows may have empty or invalid dates -> year becomes empty
    years = pd.to_datetime(df["Release_Date"], errors="coerce").dt.year
    df["year"] = years.fillna("").astype(str).replace("nan", "")

    # Not provided in dataset
    df["href"] = ""
    df["cast_str"] = ""

    # Build search_text
    df["search_text"] = (
        "Title: " + df["title"]
        + " | Plot: " + df["extract"]
        + " | Genres: " + df["genres_str"]
    ).apply(scrub_nan_tokens)

    out_df = df[["title", "year", "href", "extract", "genres_str", "cast_str", "search_text"]].copy()
    out_df = out_df[out_df["search_text"].str.len() > 30].reset_index(drop=True)

    out_df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(f"Rows: {len(out_df)} | Columns: {list(out_df.columns)}")


if __name__ == "__main__":
    preprocess_pablinho()
