# Semantic Counting in Vector Databases

A system that answers **"How many documents satisfy a semantic condition?"** by combining vector embeddings, clustering, and LLM-based filtering.

Implements a `semantic_count(*)` operator over the [STS-B dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts) using `sentence-transformers/all-MiniLM-L6-v2` for embeddings, HDBSCAN for clustering, and Google Colab's built-in LLM for semantic reasoning.

## Architecture

```
Query ("sentences about happiness")
  │
  ├─ Embed query with all-MiniLM-L6-v2
  ├─ Find top-K nearest clusters by centroid distance
  ├─ LLM filters clusters by summary relevance
  ├─ LLM checks individual documents in surviving clusters
  └─ Return final semantic count + stats
```

## Quick Start (Google Colab)

Open `notebooks/semantic_counting_demo.ipynb` in Google Colab — it clones the repo, installs dependencies, and runs the full pipeline. No API keys needed.

The first cell handles setup:
```python
!git clone https://github.com/donggunkwak/Semantic-Count.git
%cd Semantic-Count
!pip install -q -r requirements.txt
```

LLM calls use Colab's built-in AI:
```python
from google.colab import ai
response = ai.generate_text("your prompt here")
```

## Project Structure

```
├── src/
│   ├── config.py          # Paths, constants, configuration
│   ├── data_loader.py     # Load STS-B dataset
│   ├── embeddings.py      # Generate & cache sentence embeddings
│   ├── clustering.py      # HDBSCAN clustering + cluster centroids
│   ├── llm.py             # Google Colab AI wrapper (swappable)
│   ├── summarizer.py      # LLM-based cluster summarization
│   ├── query_engine.py    # Semantic counting query engine
│   └── baseline.py        # Embedding-only baseline (no LLM filtering)
├── notebooks/
│   └── semantic_counting_demo.ipynb
├── data/                  # Cached embeddings, clusters, summaries
├── outputs/               # Query results
├── run_pipeline.py        # CLI entry point (requires Colab runtime)
└── requirements.txt
```

## Pipeline Stages

| Stage | Description | Artifact |
|-------|-------------|----------|
| **Load** | Fetch STS-B from Hugging Face, deduplicate sentences | — |
| **Embed** | Encode with all-MiniLM-L6-v2 (384-dim) | `data/embeddings.npy` |
| **Cluster** | HDBSCAN over embeddings | `data/cluster_assignments.json` |
| **Summarize** | LLM summarizes sampled sentences per cluster | `data/cluster_summaries.json` |
| **Query** | Semantic count via cluster retrieval + LLM filtering | `outputs/results.json` |

## Configuration

All tunable parameters live in `src/config.py`:

- `EMBEDDING_MODEL`: sentence-transformers model name
- `HDBSCAN_MIN_CLUSTER_SIZE`: minimum cluster size
- `TOP_K_CLUSTERS`: clusters to retrieve at query time
- `SUMMARY_SAMPLE_SIZE`: sentences sampled per cluster for summarization

## Notebook

`notebooks/semantic_counting_demo.ipynb` walks through the full pipeline interactively, runs example queries ("sentences about happiness", "sentences about disagreement"), and displays intermediate results including a baseline comparison.

## Design Decisions

- **Google Colab AI**: zero-config LLM access via `google.colab.ai.generate_text()`. No API keys, no credits, no external accounts.
- **HDBSCAN over K-Means**: density-based clustering handles variable-density regions and produces a noise label (-1) for outliers, which is more realistic for semantic grouping.
- **Two-stage LLM filtering**: first filter at the cluster level (cheap — one call per cluster summary), then at the document level (expensive — one call per doc). This reduces total LLM calls significantly.
- **Caching**: embeddings, cluster assignments, and cluster summaries are all saved to disk. Re-running the pipeline skips completed stages.
