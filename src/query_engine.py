"""Semantic counting query engine.

Given a natural-language query, returns the number of documents in the
dataset that semantically satisfy the condition.

Pipeline:
  1. Embed the query.
  2. Retrieve top-K nearest clusters (by centroid cosine similarity).
  3. LLM filters clusters whose summary is relevant to the query.
  4. LLM checks individual documents in surviving clusters.
  5. Return count + diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.clustering import compute_cluster_centroids, get_cluster_members
from src.config import EMBEDDING_MODEL, RESULTS_PATH, TOP_K_CLUSTERS
from src.llm import yes_no

# ── Prompts ────────────────────────────────────────────────────────────────────

_CLUSTER_RELEVANCE_PROMPT = (
    'A user is looking for: "{query}"\n\n'
    "Here is a summary of a document cluster:\n"
    '"{summary}"\n\n'
    "Could this cluster contain documents that satisfy the user\'s query? "
    "Answer YES or NO."
)

_DOC_RELEVANCE_PROMPT = (
    'A user is looking for: "{query}"\n\n'
    "Does the following sentence satisfy the query?\n"
    '"{sentence}"\n\n'
    "Answer YES or NO."
)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class SemanticCountResult:
    query: str
    total_documents: int
    clusters_retrieved: int
    clusters_after_llm_filter: int
    documents_checked: int
    documents_matched: int
    matched_sentences: list[str] = field(default_factory=list)
    cluster_details: list[dict] = field(default_factory=list)

    @property
    def semantic_count(self) -> int:
        return self.documents_matched


# ── Engine ────────────────────────────────────────────────────────────────────


def semantic_count(
    query: str,
    sentences: list[str],
    embeddings: np.ndarray,
    labels: list[int],
    summaries: dict[str, str],
    *,
    top_k: int = TOP_K_CLUSTERS,
    model_name: str = EMBEDDING_MODEL,
    save_path: Path | None = RESULTS_PATH,
) -> SemanticCountResult:
    """Run the full semantic-counting pipeline for *query*."""

    # 1. Embed the query
    model = SentenceTransformer(model_name)
    query_vec = model.encode([query], convert_to_numpy=True)[0]

    # 2. Retrieve top-K clusters by cosine similarity to centroids
    centroids = compute_cluster_centroids(embeddings, labels)
    centroid_ids = list(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in centroid_ids])

    sims = _cosine_similarity(query_vec, centroid_matrix)
    top_indices = np.argsort(sims)[::-1][:top_k]
    retrieved_ids = [centroid_ids[i] for i in top_indices]

    print(f"[query] Retrieved {len(retrieved_ids)} clusters: {retrieved_ids}")

    # 3. LLM-based cluster filtering
    relevant_ids: list[int] = []
    cluster_details: list[dict] = []
    for cid in retrieved_ids:
        summary = summaries.get(str(cid), "")
        prompt = _CLUSTER_RELEVANCE_PROMPT.format(query=query, summary=summary)
        is_relevant = yes_no(prompt)
        cluster_details.append(
            {"cluster_id": cid, "summary": summary, "relevant": is_relevant}
        )
        if is_relevant:
            relevant_ids.append(cid)

    print(
        f"[query] {len(relevant_ids)}/{len(retrieved_ids)} clusters "
        "passed LLM relevance filter"
    )

    # 4. Check individual documents in surviving clusters
    members = get_cluster_members(labels)
    docs_checked = 0
    matched_sentences: list[str] = []

    for cid in relevant_ids:
        indices = members.get(cid, [])
        for idx in tqdm(indices, desc=f"Cluster {cid}", leave=False):
            sentence = sentences[idx]
            prompt = _DOC_RELEVANCE_PROMPT.format(query=query, sentence=sentence)
            if yes_no(prompt):
                matched_sentences.append(sentence)
            docs_checked += 1

    result = SemanticCountResult(
        query=query,
        total_documents=len(sentences),
        clusters_retrieved=len(retrieved_ids),
        clusters_after_llm_filter=len(relevant_ids),
        documents_checked=docs_checked,
        documents_matched=len(matched_sentences),
        matched_sentences=matched_sentences,
        cluster_details=cluster_details,
    )

    print(
        f"[query] semantic_count = {result.semantic_count} "
        f"({docs_checked} docs checked)"
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        _append_result(save_path, result)
        print(f"[query] Results appended to {save_path}")

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single vector and each row of a matrix."""
    vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
    mat_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return mat_norms @ vec_norm


def _append_result(path: Path, result: SemanticCountResult) -> None:
    """Append a result to the JSON results file (list of dicts)."""
    existing: list[dict] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.append(asdict(result))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
