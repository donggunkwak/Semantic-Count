"""Generate and cache per-cluster semantic summaries via LLM."""

from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

from src.clustering import get_cluster_members
from src.config import CLUSTER_SUMMARIES_PATH, SUMMARY_SAMPLE_SIZE
from src.llm import chat

_SUMMARIZE_SYSTEM = (
    "You are an expert at summarising groups of sentences. "
    "Given a sample of sentences from a cluster, produce a single concise "
    "summary (1–2 sentences) capturing the dominant semantic theme."
)

_SUMMARIZE_TEMPLATE = (
    "Below are {n} representative sentences from a cluster.\n\n"
    "{sentences}\n\n"
    "Write a concise 1–2 sentence summary of the dominant theme."
)


def summarize_clusters(
    sentences: list[str],
    labels: list[int],
    *,
    sample_size: int = SUMMARY_SAMPLE_SIZE,
    cache_path: Path = CLUSTER_SUMMARIES_PATH,
    seed: int = 42,
) -> dict[str, str]:
    """Return {cluster_id_str: summary_text}, cached to *cache_path*."""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            summaries: dict[str, str] = json.load(f)
        print(f"[summarizer] Loaded {len(summaries)} cached cluster summaries")
        return summaries

    rng = random.Random(seed)
    members = get_cluster_members(labels)
    summaries: dict[str, str] = {}

    print(f"[summarizer] Summarising {len(members)} clusters …")
    for cid in tqdm(sorted(members), desc="Summarising clusters"):
        indices = members[cid]
        sample_indices = (
            rng.sample(indices, sample_size)
            if len(indices) > sample_size
            else indices
        )
        sampled = [sentences[i] for i in sample_indices]
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sampled))
        prompt = _SUMMARIZE_TEMPLATE.format(n=len(sampled), sentences=numbered)
        summary = chat(prompt, system=_SUMMARIZE_SYSTEM)
        summaries[str(cid)] = summary

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"[summarizer] Saved summaries to {cache_path}")
    return summaries
