"""Generate and cache sentence embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDINGS_PATH


def generate_embeddings(
    sentences: list[str],
    model_name: str = EMBEDDING_MODEL,
    cache_path: Path = EMBEDDINGS_PATH,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode *sentences* and return an (N, D) float32 array.

    Results are cached to *cache_path*; a cached file is reused when its
    row count matches ``len(sentences)``.
    """
    if cache_path.exists():
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(sentences):
            print(
                f"[embeddings] Loaded cached embeddings "
                f"{embeddings.shape} from {cache_path}"
            )
            return embeddings
        print("[embeddings] Cache shape mismatch — regenerating.")

    print(f"[embeddings] Encoding {len(sentences)} sentences with {model_name} …")
    model = SentenceTransformer(model_name)
    embeddings: np.ndarray = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"[embeddings] Saved embeddings {embeddings.shape} to {cache_path}")
    return embeddings
