"""Embedding-only baseline: count by cosine similarity threshold (no LLM)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


@dataclass
class BaselineResult:
    query: str
    total_documents: int
    threshold: float
    documents_matched: int
    matched_sentences: list[str] = field(default_factory=list)

    @property
    def semantic_count(self) -> int:
        return self.documents_matched


def baseline_count(
    query: str,
    sentences: list[str],
    embeddings: np.ndarray,
    *,
    threshold: float = 0.45,
    model_name: str = EMBEDDING_MODEL,
) -> BaselineResult:
    """Count documents whose cosine similarity to *query* exceeds *threshold*.

    This is a simple baseline that uses no LLM calls — only embedding
    distance — so it is fast but less semantically precise.
    """
    model = SentenceTransformer(model_name)
    query_vec = model.encode([query], convert_to_numpy=True)[0]

    # Cosine similarities
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    sims = normed @ query_norm

    mask = sims >= threshold
    matched = [sentences[i] for i in np.where(mask)[0]]

    result = BaselineResult(
        query=query,
        total_documents=len(sentences),
        threshold=threshold,
        documents_matched=len(matched),
        matched_sentences=matched,
    )
    print(
        f"[baseline] semantic_count = {result.semantic_count} "
        f"(threshold={threshold})"
    )
    return result
