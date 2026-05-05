"""Load the Banking77 dataset and extract unique sentences."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.config import SENTENCES_PATH


def load_banking77_sentences(cache_path: Path = SENTENCES_PATH) -> list[str]:
    """Return deduplicated sentences from mteb/banking77 (train + test).

    Sentences are cached to *cache_path* so subsequent calls skip the
    Hugging Face download.
    """
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            sentences: list[str] = json.load(f)
        print(f"[data_loader] Loaded {len(sentences)} cached sentences from {cache_path}")
        return sentences

    print("[data_loader] Downloading Banking77 dataset …")
    ds = load_dataset("mteb/banking77", split="train+test")

    seen: set[str] = set()
    sentences = []
    for row in tqdm(ds, desc="Extracting sentences"):
        s = row["text"].strip()
        if s and s not in seen:
            seen.add(s)
            sentences.append(s)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)

    print(f"[data_loader] Saved {len(sentences)} unique sentences to {cache_path}")
    return sentences
