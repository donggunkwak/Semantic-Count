"""Load the Banking77 dataset and extract unique sentences."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.config import DATA_DIR, SENTENCES_PATH

LABELS_PATH = DATA_DIR / "labels.json"
LABEL_NAMES_PATH = DATA_DIR / "label_names.json"



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


def load_banking77_with_labels(
    sentences_path: Path = SENTENCES_PATH,
    labels_path: Path = LABELS_PATH,
    label_names_path: Path = LABEL_NAMES_PATH,
) -> tuple[list[str], list[int], list[str]]:
    """Return (sentences, numeric_labels, label_texts).

    - sentences: deduplicated sentence list (same as load_banking77_sentences)
    - numeric_labels: parallel list of Banking77 integer labels per sentence
    - label_texts: parallel list of label text strings per sentence
    """
    if sentences_path.exists() and labels_path.exists() and label_names_path.exists():
        with open(sentences_path, "r", encoding="utf-8") as f:
            sentences: list[str] = json.load(f)
        with open(labels_path, "r", encoding="utf-8") as f:
            numeric_labels: list[int] = json.load(f)
        with open(label_names_path, "r", encoding="utf-8") as f:
            label_texts: list[str] = json.load(f)
        print(f"[data_loader] Loaded {len(sentences)} sentences with labels from cache")
        return sentences, numeric_labels, label_texts

    print("[data_loader] Downloading Banking77 dataset …")
    ds = load_dataset("mteb/banking77", split="train+test")

    seen: dict[str, int] = {}
    sentences = []
    numeric_labels = []
    label_texts = []
    for row in tqdm(ds, desc="Extracting sentences"):
        s = row["text"].strip()
        if s and s not in seen:
            seen[s] = row["label"]
            sentences.append(s)
            numeric_labels.append(row["label"])
            label_texts.append(row["label_text"])

    sentences_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sentences_path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(numeric_labels, f)
    with open(label_names_path, "w", encoding="utf-8") as f:
        json.dump(label_texts, f, ensure_ascii=False, indent=2)

    print(f"[data_loader] Saved {len(sentences)} sentences + labels")
    return sentences, numeric_labels, label_texts
