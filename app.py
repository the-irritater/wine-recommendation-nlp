"""
Streamlit Web Application for Wine Recommendation NLP System
==============================================================
Provides interactive natural language search interface for wine selection.

Usage:
    streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import nmslib

st.set_page_config(
    page_title="Wine Recommendation System",
    page_icon="🍷",
    layout="wide",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "winemag-data-130k-v2.csv")


@st.cache_resource
def load_model():
    """Load SentenceTransformer embedding model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def load_dataset(data_path: str = DATA_PATH, sample_size: int = 5000):
    """Load and sample preprocessed wine review dataset."""
    if not os.path.exists(data_path):
        st.error(f"Dataset file not found at {data_path}")
        return pd.DataFrame()

    df = pd.read_csv(data_path).drop_duplicates("description").dropna(subset=["price"])
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return df


@st.cache_resource
def build_index(_subset_df, _model):
    """Build NMSLIB vector search index."""
    embeddings = _model.encode(_subset_df["description"].tolist(), convert_to_numpy=True)
    index = nmslib.init(method="hnsw", space="cosinesimil")
    index.addDataPointBatch(embeddings)
    index.createIndex({"post": 2}, print_progress=False)
    return index, embeddings


def main():
    st.title("Wine Recommendation System")
    st.subheader("Semantic Search Engine for Wine Discovery")

    df = load_dataset()
    if df.empty:
        st.stop()

    model = load_model()
    index, embeddings = build_index(df, model)

    # Sidebar Controls
    st.sidebar.header("Search & Filtering Options")
    max_price = st.sidebar.slider("Maximum Price ($)", min_value=5, max_value=200, value=50, step=5)
    min_points = st.sidebar.slider("Minimum Rating (Points)", min_value=80, max_value=100, value=85, step=1)
    top_k = st.sidebar.number_input("Number of Recommendations", min_value=1, max_value=20, value=5)

    # Search Bar
    query = st.text_input(
        "Describe your preferred wine profile:",
        value="dry and fruity red with plum and oak notes",
        placeholder="e.g. crisp white wine with high acidity and citrus",
    )

    if st.button("Search Recommendations") or query:
        with st.spinner("Searching semantic embedding space..."):
            query_vec = model.encode([query], convert_to_numpy=True)
            ids, distances = index.knnQuery(query_vec, k=top_k * 3)

            results = []
            for idx, dist in zip(ids, distances):
                row = df.iloc[idx]
                if row["price"] <= max_price and row["points"] >= min_points:
                    results.append({
                        "Title": row["title"],
                        "Variety": row["variety"],
                        "Country": row["country"],
                        "Price ($)": float(row["price"]),
                        "Points": int(row["points"]),
                        "Distance": round(float(dist), 4),
                        "Description": row["description"],
                    })
                if len(results) >= top_k:
                    break

            res_df = pd.DataFrame(results)

            if not res_df.empty:
                st.success(f"Found {len(res_df)} matching recommendations:")
                for idx, row in res_df.iterrows():
                    with st.expander(f"{row['Title']} - ${row['Price ($)']} ({row['Points']} Points)"):
                        st.write(f"**Variety:** {row['Variety']} | **Country:** {row['Country']}")
                        st.write(f"**Similarity Distance:** {row['Distance']}")
                        st.write(f"**Review Description:** {row['Description']}")

                st.subheader("Results Table")
                st.dataframe(res_df[["Title", "Variety", "Price ($)", "Points", "Distance"]], use_container_width=True)
            else:
                st.warning("No wines matched the current price and point filter criteria. Try expanding filter boundaries.")


if __name__ == "__main__":
    main()
