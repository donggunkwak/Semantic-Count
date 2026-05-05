"""HDBSCAN clustering over sentence embeddings.

Uses UMAP to reduce dimensionality before clustering for performance on
large datasets (O(n²) pairwise distances become tractable at ~25 dims).
"""

from __future__ import annotations

import json
from pathlib import Path

import hdbscan
import numpy as np
import umap

from src.config import (
    CLUSTER_ASSIGNMENTS_PATH,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
)

_UMAP_N_COMPONENTS = 25


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    cache_path: Path = CLUSTER_ASSIGNMENTS_PATH,
) -> list[int]:
    """Run UMAP + HDBSCAN and return per-document cluster labels (-1 = noise).

    Results are cached to *cache_path*.
    """
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            labels: list[int] = json.load(f)
        if len(labels) == embeddings.shape[0]:
            n_clusters = len(set(labels) - {-1})
            n_noise = labels.count(-1)
            print(
                f"[clustering] Loaded cached labels: "
                f"{n_clusters} clusters, {n_noise} noise points"
            )
            return labels
        print("[clustering] Cache size mismatch — reclustering.")

    print(f"[clustering] Reducing {embeddings.shape[1]}d → {_UMAP_N_COMPONENTS}d with UMAP …")
    reducer = umap.UMAP(
        n_components=_UMAP_N_COMPONENTS,
        metric="cosine",
        n_jobs=-1,
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)

    print(
        f"[clustering] Running HDBSCAN "
        f"(min_cluster_size={min_cluster_size}, min_samples={min_samples}) …"
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    clusterer.fit(reduced)
    labels = clusterer.labels_.tolist()

    n_clusters = len(set(labels) - {-1})
    n_noise = labels.count(-1)
    print(f"[clustering] Found {n_clusters} clusters, {n_noise} noise points")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(labels, f)

    print(f"[clustering] Saved cluster assignments to {cache_path}")
    return labels


def compute_cluster_centroids(
    embeddings: np.ndarray,
    labels: list[int],
) -> dict[int, np.ndarray]:
    """Return {cluster_id: centroid_vector} (excludes noise label -1)."""
    centroids: dict[int, np.ndarray] = {}
    labels_arr = np.array(labels)
    for cid in sorted(set(labels) - {-1}):
        mask = labels_arr == cid
        centroids[cid] = embeddings[mask].mean(axis=0)
    return centroids


def get_cluster_members(
    labels: list[int],
) -> dict[int, list[int]]:
    """Return {cluster_id: [doc_indices]} (excludes noise label -1)."""
    members: dict[int, list[int]] = {}
    for idx, cid in enumerate(labels):
        if cid == -1:
            continue
        members.setdefault(cid, []).append(idx)
    return members
