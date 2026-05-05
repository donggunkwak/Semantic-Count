#!/usr/bin/env python
"""CLI entry point: run the full semantic-counting pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from src.baseline import baseline_count
from src.clustering import cluster_embeddings
from src.config import RESULTS_PATH
from src.data_loader import load_banking77_sentences
from src.embeddings import generate_embeddings
from src.query_engine import semantic_count
from src.summarizer import summarize_clusters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic Counting in Vector Databases"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="I want to transfer money to another account",
        help="Natural-language query for semantic counting",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing (assume cached artifacts exist)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the embedding baseline (no LLM calls)",
    )
    parser.add_argument(
        "--baseline-threshold",
        type=float,
        default=0.45,
        help="Cosine-similarity threshold for the baseline",
    )
    args = parser.parse_args()

    # ── Preprocessing ─────────────────────────────────────────────────────
    print("=" * 60)
    print("STAGE 1 — Load data")
    print("=" * 60)
    sentences = load_banking77_sentences()

    print("\n" + "=" * 60)
    print("STAGE 2 — Generate embeddings")
    print("=" * 60)
    embeddings = generate_embeddings(sentences)

    print("\n" + "=" * 60)
    print("STAGE 3 — Cluster embeddings")
    print("=" * 60)
    labels = cluster_embeddings(embeddings)

    if not args.baseline_only:
        print("\n" + "=" * 60)
        print("STAGE 4 — Summarise clusters (LLM)")
        print("=" * 60)
        summaries = summarize_clusters(sentences, labels)

    # ── Query ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f'STAGE 5 — Semantic count: "{args.query}"')
    print("=" * 60)

    if args.baseline_only:
        result = baseline_count(
            args.query, sentences, embeddings, threshold=args.baseline_threshold
        )
    else:
        result = semantic_count(args.query, sentences, embeddings, labels, summaries)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
