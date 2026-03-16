# Wine Recommendation System using NLP and Semantic Search

## Project Overview
This project builds a wine recommendation system using Natural Language Processing (NLP). It analyzes wine review descriptions, converts them into vector embeddings using Sentence Transformers, and retrieves the most semantically similar wines based on user queries such as "dry and fruity" or "high acidity taste tart and zesty".

## Objectives
- Perform data cleaning and exploratory data analysis on a wine reviews dataset
- Generate semantic embeddings from wine descriptions
- Build an approximate nearest neighbor search system using NMSLIB
- Recommend wines based on natural language input
- Visualize wine description embeddings using t-SNE

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Sentence Transformers
- NMSLIB
- Scikit-learn

## Dataset
This project uses the Wine Reviews dataset containing approximately 130k wine reviews with attributes such as:
- country
- winery
- title
- variety
- description
- price
- points

## Project Workflow
1. Load and clean the wine dataset
2. Remove duplicate descriptions and missing price values
3. Perform exploratory data analysis with charts
4. Generate sentence embeddings from wine descriptions
5. Build a semantic similarity search index using NMSLIB
6. Retrieve top matching wines for custom text queries
7. Visualize embeddings using t-SNE

## Example Queries
- dry and fruity
- high acidity taste tart and zesty

## Output
The system returns the most similar wines based on the meaning of the query, not just keyword matching.

## Project Structure
```text
wine-recommendation-nlp/
│
├── data/
│   └── winemag-data-130k-v2.csv
├── outputs/
│   ├── top_countries.png
│   ├── top_varieties.png
│   ├── most_expensive_varieties.png
│   ├── highest_quality_varieties.png
│   ├── best_value_wines.png
│   └── wine_tsne_visualization.png
├── wine_recommendation_nlp.py
├── requirements.txt
└── README.md
```
## Installation

Clone the repository and install dependencies:

git clone https://github.com/your-username/wine-recommendation-nlp.git
cd wine-recommendation-nlp
pip install -r requirements.txt

## Run the Project

python wine_recommendation_nlp.py

## Key Features

- Natural language wine search
- Semantic recommendation engine
- EDA visualizations
- Embedding-based similarity matching
- t-SNE visualization for semantic clustering

- ## Future Improvements

- Build a Streamlit web app interface
- Add filters for country, price, and wine variety
- Improve recommendation accuracy using larger embedding models
- Deploy as an interactive project

- ## Author
Sanman Kadam  
GitHub: https://github.com/the-irritater
