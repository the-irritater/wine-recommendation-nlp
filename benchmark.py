"""
Search Engine Benchmark Module for Wine Recommendation NLP
=============================================================
Benchmarks latency (ms) and precision/recall between:
1. NMSLIB (HNSW Cosine Index)
2. Brute-Force Cosine Similarity
3. TF-IDF Keyword Baseline
"""

import os
import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import nmslib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "winemag-data-130k-v2.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_benchmark():
    print("=" * 60)
    print("SEARCH ENGINE BENCHMARK: NMSLIB vs BRUTE-FORCE vs TF-IDF")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}. Skipping benchmark.")
        return

    df = pd.read_csv(DATA_PATH).drop_duplicates("description").dropna(subset=["price"])
    sample = df.sample(n=2000, random_state=42).reset_index(drop=True)

    print(f"Dataset benchmark sample: {len(sample):,} wine reviews.")

    # 1. Generate Embeddings & TF-IDF
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sample["description"].tolist(), convert_to_numpy=True)

    tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(sample["description"])

    # 2. Build NMSLIB Index
    nms_index = nmslib.init(method="hnsw", space="cosinesimil")
    nms_index.addDataPointBatch(embeddings)
    nms_index.createIndex({"post": 2}, print_progress=False)

    test_queries = [
        "dry and fruity red",
        "high acidity taste tart and zesty",
        "crisp white wine with citrus notes",
        "bold oaky cabernet sauvignon",
        "sweet dessert wine with honey notes",
    ]

    results = []

    for q in test_queries:
        # NMSLIB timing
        t0 = time.time()
        q_emb = model.encode([q], convert_to_numpy=True)
        nms_ids, nms_dists = nms_index.knnQuery(q_emb, k=10)
        t_nms = (time.time() - t0) * 1000

        # Brute-force Cosine timing
        t0 = time.time()
        sims = cosine_similarity(q_emb, embeddings)[0]
        bf_ids = np.argsort(sims)[::-1][:10]
        t_bf = (time.time() - t0) * 1000

        # TF-IDF timing
        t0 = time.time()
        q_tf = tfidf.transform([q])
        tf_sims = cosine_similarity(q_tf, tfidf_matrix)[0]
        tf_ids = np.argsort(tf_sims)[::-1][:10]
        t_tf = (time.time() - t0) * 1000

        # Measure Recall@10 of NMSLIB vs Brute-force ground truth
        overlap_nms = len(set(nms_ids).intersection(set(bf_ids)))
        recall_nms = (overlap_nms / 10.0) * 100

        overlap_tf = len(set(tf_ids).intersection(set(bf_ids)))
        recall_tf = (overlap_tf / 10.0) * 100

        results.append({
            "query": q,
            "nmslib_latency_ms": round(t_nms, 2),
            "bruteforce_latency_ms": round(t_bf, 2),
            "tfidf_latency_ms": round(t_tf, 2),
            "nmslib_recall_at_10_pct": recall_nms,
            "tfidf_recall_at_10_pct": recall_tf,
        })

    res_df = pd.DataFrame(results)

    print("\nBenchmark Summary Statistics:")
    print(f"Mean NMSLIB Latency: {res_df['nmslib_latency_ms'].mean():.2f} ms")
    print(f"Mean Brute-Force Latency: {res_df['bruteforce_latency_ms'].mean():.2f} ms")
    print(f"Mean TF-IDF Latency: {res_df['tfidf_latency_ms'].mean():.2f} ms")
    print(f"Mean NMSLIB Recall@10 vs Ground Truth: {res_df['nmslib_recall_at_10_pct'].mean():.1f}%")
    print(f"Mean TF-IDF Recall@10 vs Ground Truth: {res_df['tfidf_recall_at_10_pct'].mean():.1f}%")

    out_file = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    res_df.to_csv(out_file, index=False)
    print(f"Saved benchmark results to {out_file}")


if __name__ == "__main__":
    run_benchmark()
