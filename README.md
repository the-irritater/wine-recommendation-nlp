# Wine Recommendation System: Natural Language Processing and Semantic Search

Developing a wine recommendation engine utilizing Transformer embeddings and approximate nearest neighbor search to match natural language consumer preferences against expert wine review descriptions.

## Problem Statement

Traditional wine search relies on exact keyword matching or rigid categorical filters (e.g., region, price range). However, wine taste preferences are inherently sensory and descriptive (e.g., "dry and fruity red with oak and plum notes").

Keyword search fails to map semantic equivalence between flavor descriptors (e.g., "tart" vs. "high acidity"). This project implements vector-based semantic search to evaluate textual descriptions against user queries in a shared embedding space.

## Query Performance and System Evaluation

### Sample Queries & Output Results

| Query | Top Recommendation | Variety | Price | Score | Distance |
|-|-|-|-|-|-|
| "dry and fruity red" | Marqués de Cáceres | Tempranillo | $12 | 87 | 0.1245 |
| "high acidity taste tart and zesty" | Kim Crawford | Sauvignon Blanc | $14 | 90 | 0.1082 |
| "bold and peppery good with steak" | Barossa Valley Shiraz | Shiraz | $22 | 91 | 0.1154 |

### Search Engine Benchmarks

| Methodology | Recall@10 (vs Ground Truth) | Query Latency | Notes |
|-|-|-|-|
| NMSLIB (HNSW Cosine Index) | 98.2% | 0.8ms | Selected production index |
| Brute-Force Cosine Similarity | 100.0% | 45.0ms | Ground truth baseline |
| TF-IDF Keyword Match | 72.1% | 12.0ms | Misses semantic synonyms |

NMSLIB HNSW indexing achieves a 36% improvement in recall compared to keyword matching while maintaining sub-millisecond query latency.

## Preprocessing and Index Architecture

1. **Dataset Preprocessing**: Cleaned 130,000 wine reviews from the WineMag dataset, removing duplicate descriptions and incomplete price fields.
2. **Embedding Generation**: Encoded text descriptions into vector representations using the `all-MiniLM-L6-v2` SentenceTransformer architecture.
3. **Index Construction**: Configured an NMSLIB HNSW index optimized for cosine similarity.
4. **Dimensionality Reduction**: Visualized embedding distribution using 2D t-SNE projections.

## Project Structure

```
wine-recommendation-nlp/
├── data/
│   └── winemag-data-130k-v2.csv
├── outputs/
│   ├── benchmark_results.csv
│   └── wine_tsne_visualization.png
├── app.py
├── benchmark.py
├── wine_recommendation_nlp.py
├── requirements.txt
└── README.md
```

## How to Run

### Execute Recommendation Script
```bash
pip install -r requirements.txt
python wine_recommendation_nlp.py
```

### Run Benchmark Analysis
```bash
python benchmark.py
```

### Launch Streamlit Interface
```bash
streamlit run app.py
```

## Author

Sanman Kadam  
MSc Statistics | Data Analyst
