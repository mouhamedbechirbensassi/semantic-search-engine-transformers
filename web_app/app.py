from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.config import OUTPUTS_DIR, DEFAULT_EMBEDDING_MODEL

st.set_page_config(page_title="Semantic Movie Search", layout="wide")

EMB_DIR = OUTPUTS_DIR / "embeddings"
INDEX_DIR = OUTPUTS_DIR / "indexes"
INDEX_PATH = INDEX_DIR / "movies_faiss.index"
META_PATH = EMB_DIR / "movies_metadata.parquet"


@st.cache_resource
def load_model():
    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)


@st.cache_resource
def load_index():
    return faiss.read_index(str(INDEX_PATH))


@st.cache_data
def load_meta():
    df = pd.read_parquet(META_PATH)
    # year may be string; keep a numeric version for filtering
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def snippet(text: str, n: int = 240) -> str:
    t = str(text).replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 3] + "..."


def search(query: str, top_k: int):
    model = load_model()
    index = load_index()
    meta = load_meta()

    qvec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, ids = index.search(qvec, top_k)

    res = meta.iloc[ids[0]].copy()
    res.insert(0, "score", scores[0])
    return res.reset_index(drop=True)


st.title("🎬 Semantic Movie Search")
st.caption("Semantic search over a merged dataset (Wikipedia + modern movie dataset).")

# ---- Sidebar filters
meta = load_meta()

with st.sidebar:
    st.header("Filters")

    source_options = ["all"] + sorted([s for s in meta["source"].dropna().unique().tolist() if str(s).strip() != ""])
    source_choice = st.selectbox("Source", source_options, index=0)

    y_min = int(meta["year_num"].min()) if meta["year_num"].notna().any() else 1900
    y_max = int(meta["year_num"].max()) if meta["year_num"].notna().any() else 2026
    year_range = st.slider("Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max))

    genre_keyword = st.text_input("Genre contains (optional)", value="").strip().lower()

top_k = st.slider("Top K results", min_value=3, max_value=30, value=10)
query = st.text_input("Search query", value="")

if st.button("Search") and query.strip():
    results = search(query.strip(), top_k=top_k)

    # Apply filters AFTER retrieval (fast + simple)
    if source_choice != "all":
        results = results[results["source"] == source_choice]

    lo, hi = year_range
    results = results[(results["year_num"].isna()) | ((results["year_num"] >= lo) & (results["year_num"] <= hi))]

    if genre_keyword:
        results = results[results["genres_str"].astype(str).str.lower().str.contains(genre_keyword, na=False)]

    if results.empty:
        st.warning("No results after filters. Try widening year range or removing filters.")
    else:
        for i, row in results.head(top_k).iterrows():
            with st.container(border=True):
                st.markdown(f"### #{i+1} — {row['title']} ({row.get('year','')})")
                st.markdown(f"**Score:** `{row['score']:.3f}`  |  **Source:** `{row.get('source','')}`")
                st.markdown(f"**Genres:** {row.get('genres_str','')}")
                if str(row.get("cast_str","")).strip():
                    st.markdown(f"**Cast:** {row.get('cast_str','')}")
                st.markdown(f"**Plot:** {snippet(row.get('extract',''))}")
                href = str(row.get("href", "")).strip()
                if href:
                    st.markdown(f"**Link:** {href}")

else:
    st.info("Enter a query and click Search.")
