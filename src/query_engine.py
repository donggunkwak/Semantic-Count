"""Semantic counting query engine.

Given a natural-language query, returns the number of documents in the
dataset that semantically satisfy the condition.

Pipeline:
  1. Embed the query.
  2. Retrieve top-K nearest clusters (by centroid cosine similarity).
  3. LLM filters clusters whose summary is relevant to the query.
  4. LLM checks individual documents (yes/no + similarity score 0–1).
  5. Rank matched sentences by score, return count + ranked list.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.clustering import compute_cluster_centroids, get_cluster_members
from src.config import EMBEDDING_MODEL, OUTPUT_DIR, RESULTS_PATH, TOP_K_CLUSTERS
from src.llm import yes_no, yes_no_score

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
    "Answer YES or NO, followed by a similarity score from 0.0 to 1.0 "
    "(where 1.0 means the sentence is exactly the same as the query). "
    "Format: YES 0.85 or NO 0.15"
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
    scored_sentences: list[dict] = field(default_factory=list)
    cluster_details: list[dict] = field(default_factory=list)

    @property
    def semantic_count(self) -> int:
        return self.documents_matched

    @property
    def matched_sentences(self) -> list[str]:
        return [entry["sentence"] for entry in self.scored_sentences]

    @property
    def ranked_sentences(self) -> list[dict]:
        """Matched sentences sorted by similarity score (highest first)."""
        return sorted(self.scored_sentences, key=lambda x: x["score"], reverse=True)


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
    save_ranked_txt: bool = True,
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

    # 4. Check individual documents — get yes/no + similarity score
    members = get_cluster_members(labels)
    docs_checked = 0
    scored_sentences: list[dict] = []

    for cid in relevant_ids:
        indices = members.get(cid, [])
        for idx in tqdm(indices, desc=f"Cluster {cid}", leave=False):
            sentence = sentences[idx]
            prompt = _DOC_RELEVANCE_PROMPT.format(query=query, sentence=sentence)
            is_relevant, score = yes_no_score(prompt)
            if is_relevant:
                scored_sentences.append({"sentence": sentence, "score": score})
            docs_checked += 1

    scored_sentences.sort(key=lambda x: x["score"], reverse=True)

    result = SemanticCountResult(
        query=query,
        total_documents=len(sentences),
        clusters_retrieved=len(retrieved_ids),
        clusters_after_llm_filter=len(relevant_ids),
        documents_checked=docs_checked,
        documents_matched=len(scored_sentences),
        scored_sentences=scored_sentences,
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

    if save_ranked_txt:
        txt_path = _save_ranked_txt(result)
        print(f"[query] Ranked results saved to {txt_path}")

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


def _save_ranked_txt(result: SemanticCountResult) -> Path:
    """Write ranked matched sentences to a .txt file in the outputs dir."""
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in result.query)
    safe_name = safe_name.strip().replace(" ", "_")[:80]
    txt_path = OUTPUT_DIR / f"ranked_{safe_name}.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Query: {result.query}\n")
        f.write(f"Total matched sentences: {result.semantic_count}\n")
        f.write("=" * 70 + "\n\n")
        for rank, entry in enumerate(result.ranked_sentences, start=1):
            f.write(f"{rank:>4}. [score={entry['score']:.4f}] {entry['sentence']}\n")

    return txt_path
