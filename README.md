# Multilingual SBERT Embeddings with t-SNE and K-Means Clustering

> Course: Natural Language Processing · Semester 2

## Overview

Semantic clustering of 148 short Spanish sentences. The notebook cleans the corpus, encodes each sentence with a multilingual Sentence-BERT model, projects the embeddings to two dimensions with t-SNE, groups them with K-Means and interprets the resulting domains. A second, document-only assignment on news topic clustering with FastText embeddings and density-based clustering is included as complementary reading.

## Repository structure

- `notebooks/sbert-embeddings-clustering.ipynb` - complete executed notebook.
- `assets/tsne-sbert-clusters.png` - labeled 2D cluster plot produced by the notebook.
- `docs/fasttext-clustering-report.pdf` - report comparing K-Means and DBSCAN on FastText embeddings.

## Method

- Lowercasing, regex cleanup `[^a-záéíóúñü\s]` and whitespace collapsing.
- `SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')`.
- `TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)`.
- `KMeans(n_clusters=4, random_state=42)` and a labeled scatter plot with seaborn.
- Exports position CSV and the PNG figure.

## Results

Four clusters of sizes **48, 38, 38 and 24**, interpreted as political/democratic discourse, sport and health, and technology/computing topics.

## Data

The notebook expects a `emdeddings_dataset.csv` file with a `frase` column. The source CSV was not included in the original folder, so add your own 148-sentence CSV (or adapt the loader) before running. The model is downloaded on first use.

## Requirements

`sentence-transformers`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `nltk` and `jupyter`.

## How to run

```bash
pip install -r requirements.txt
# place emdeddings_dataset.csv in the repository root
jupyter notebook notebooks/sbert-embeddings-clustering.ipynb
```

---
Author: Karla Maria Monroy Alvarez
