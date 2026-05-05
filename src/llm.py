"""Google Colab AI wrapper — uses Colab's built-in LLM, no API keys needed."""

from __future__ import annotations

import re
import time

from google.colab import ai

from src.config import LLM_MODEL

_DEFAULT_DELAY = 0.5       # seconds between LLM calls (summaries, cluster filter)
_MAX_RETRIES = 5
_RETRY_BACKOFF_BASE = 2.0  # exponential backoff multiplier


def chat(
    prompt: str,
    *,
    system: str = "You are a helpful assistant.",
    model: str = LLM_MODEL,
    delay: float | None = None,
) -> str:
    """Generate text using Colab's built-in LLM with retry + rate-limit delay.

    *delay* overrides the default pause between calls.  Pass 0 to skip.
    """
    pause = _DEFAULT_DELAY if delay is None else delay
    full_prompt = f"{system}\n\n{prompt}"
    for attempt in range(_MAX_RETRIES):
        try:
            response = ai.generate_text(full_prompt, model_name=model)
            if pause:
                time.sleep(pause)
            return response.strip()
        except Exception as exc:
            wait = _RETRY_BACKOFF_BASE ** attempt
            print(f"[llm] attempt {attempt + 1}/{_MAX_RETRIES} failed: {exc}  "
                  f"— retrying in {wait:.0f}s")
            time.sleep(wait)
    response = ai.generate_text(full_prompt, model_name=model)
    if pause:
        time.sleep(pause)
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
    delay: float = 0.2,
) -> tuple[bool, float]:
    """Ask a yes/no question and also extract a similarity score (0–1).

    Returns (is_yes, score).  Uses a shorter delay by default since this
    is called once per document in the hot loop.
    """
    answer = chat(prompt, system=system, model=model, delay=delay)
    is_yes = answer.upper().startswith("YES")
    match = re.search(r"(\d+\.\d+|\d+)", answer)
    score = float(match.group(1)) if match else (1.0 if is_yes else 0.0)
    score = max(0.0, min(1.0, score))
    return is_yes, score
