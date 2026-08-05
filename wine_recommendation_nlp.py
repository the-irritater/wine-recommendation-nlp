"""
Wine Recommendation System using NLP & Semantic Search
========================================================
Converts wine review descriptions into vector embeddings using Sentence Transformers
and retrieves semantically similar wines using NMSLIB approximate nearest neighbor search.
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import nmslib
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "winemag-data-130k-v2.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_preprocess_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load and clean the wine dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Original dataset shape: {df.shape}")

    # Remove duplicate descriptions
    df = df.drop_duplicates("description")
    print(f"Shape after dropping duplicate descriptions: {df.shape}")

    # Remove null prices
    df = df.dropna(subset=["price"])
    print(f"Shape after dropping null prices: {df.shape}")

    return df


def generate_embeddings(df: pd.DataFrame, sample_frac: float = 0.05, random_state: int = 42) -> tuple[pd.DataFrame, np.ndarray, SentenceTransformer]:
    """Generate vector embeddings from wine descriptions using SentenceTransformer."""
    subset = df.sample(frac=sample_frac, random_state=random_state).copy()
    subset.reset_index(drop=True, inplace=True)

    print(f"Sampled {len(subset):,} reviews for embedding generation.")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(subset["description"].tolist(), show_progress_bar=True, convert_to_numpy=True)
    print(f"Embedding matrix shape: {embeddings.shape}")

    subset["vectors"] = list(embeddings)
    return subset, embeddings, model


def build_nmslib_index(embeddings: np.ndarray) -> nmslib.FloatIndex:
    """Construct an NMSLIB HNSW cosine index for fast nearest neighbor search."""
    index = nmslib.init(method="hnsw", space="cosinesimil")
    index.addDataPointBatch(embeddings)
    index.createIndex({"post": 2}, print_progress=False)
    print("NMSLIB index created successfully.")
    return index


def search_wines(
    query_text: str,
    subset_df: pd.DataFrame,
    embeddings: np.ndarray,
    index: nmslib.FloatIndex,
    model: SentenceTransformer,
    top_k: int = 5,
) -> pd.DataFrame:
    """Perform semantic search for matching wine recommendations."""
    query_vec = model.encode([query_text], convert_to_numpy=True)
    ids, distances = index.knnQuery(query_vec, k=top_k)

    results = []
    for idx, dist in zip(ids, distances):
        results.append({
            "title": subset_df.at[idx, "title"],
            "variety": subset_df.at[idx, "variety"],
            "country": subset_df.at[idx, "country"],
            "price": subset_df.at[idx, "price"],
            "points": subset_df.at[idx, "points"],
            "distance": round(float(dist), 4),
            "description": subset_df.at[idx, "description"],
        })

    return pd.DataFrame(results)


def plot_tsne_clusters(subset_df: pd.DataFrame, embeddings: np.ndarray, output_dir: str = OUTPUT_DIR):
    """Generate t-SNE 2D visualization of wine description embeddings."""
    print("Calculating t-SNE 2D projections...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    projections = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    plt.scatter(projections[:, 0], projections[:, 1], alpha=0.6, c="#3498db", edgecolors="none")
    plt.title("Wine Description Embeddings t-SNE Projection")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "wine_tsne_visualization.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE visualization saved to {filepath}")


if __name__ == "__main__":
    try:
        df = load_and_preprocess_data()
        subset, embeddings, model = generate_embeddings(df, sample_frac=0.02)
        index = build_nmslib_index(embeddings)

        query1 = "dry and fruity red"
        print(f"\nTop recommendations for: '{query1}'")
        res1 = search_wines(query1, subset, embeddings, index, model, top_k=3)
        print(res1[["title", "variety", "price", "distance"]].to_string())

        query2 = "high acidity taste tart and zesty"
        print(f"\nTop recommendations for: '{query2}'")
        res2 = search_wines(query2, subset, embeddings, index, model, top_k=3)
        print(res2[["title", "variety", "price", "distance"]].to_string())

        plot_tsne_clusters(subset, embeddings)

    except Exception as e:
        print(f"Execution error: {e}")
