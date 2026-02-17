from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.config import RAW_DIR

DATASET_URL = "https://raw.githubusercontent.com/prust/wikipedia-movie-data/master/movies.json"
RAW_FILE = RAW_DIR / "movies.json"
RAW_CSV = RAW_DIR / "movies_raw.csv"


def download_dataset(url: str = DATASET_URL, out_path: Path = RAW_FILE) -> Path:
    """
    Download the JSON dataset and save to data/raw/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(url)  # loads JSON directly from URL
    df.to_json(out_path, orient="records", force_ascii=False)
    df.to_csv(RAW_CSV, index=False)
    print(f"Saved JSON to: {out_path}")
    print(f"Saved CSV  to: {RAW_CSV}")
    print(f"Rows: {len(df)} | Columns: {list(df.columns)}")
    return out_path


if __name__ == "__main__":
    download_dataset()
