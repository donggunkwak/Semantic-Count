"""Google Colab AI wrapper — uses Colab's built-in LLM, no API keys needed."""

from __future__ import annotations

import re

from google.colab import ai

from src.config import LLM_MODEL


def chat(
    prompt: str,
    *,
    system: str = "You are a helpful assistant.",
    model: str = LLM_MODEL,
) -> str:
    """Generate text using Colab's built-in LLM."""
    full_prompt = f"{system}\n\n{prompt}"
    response = ai.generate_text(full_prompt, model_name=model)
    return response.strip()


def yes_no(
    prompt: str,
    *,
    system: str = "You are a helpful assistant. Answer only YES or NO.",
    model: str = LLM_MODEL,
) -> bool:
    """Ask a yes/no question and return True for YES."""
    answer = chat(prompt, system=system, model=model)
    return answer.upper().startswith("YES")


def yes_no_score(
    prompt: str,
    *,
    system: str = (
        "You are a helpful assistant. "
        "Answer with YES or NO followed by a similarity score from 0.0 to 1.0. "
        "Format: YES 0.85 or NO 0.15"
    ),
    model: str = LLM_MODEL,
) -> tuple[bool, float]:
    """Ask a yes/no question and also extract a similarity score (0–1).

    Returns (is_yes, score).  Falls back to score=0.0 on parse failure.
    """
    answer = chat(prompt, system=system, model=model)
    is_yes = answer.upper().startswith("YES")
    match = re.search(r"(\d+\.\d+|\d+)", answer)
    score = float(match.group(1)) if match else (1.0 if is_yes else 0.0)
    score = max(0.0, min(1.0, score))
    return is_yes, score
