"""Ollama local LLM wrapper — no API keys required.

Requires Ollama to be installed and running locally.
Install: https://ollama.com
Pull a model: `ollama pull llama3.2`
"""

from __future__ import annotations

import ollama

from src.config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


def chat(
    prompt: str,
    *,
    system: str = "You are a helpful assistant.",
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """Send a single-turn chat completion and return the assistant reply."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )
    return response["message"]["content"].strip()


def yes_no(
    prompt: str,
    *,
    system: str = "You are a helpful assistant. Answer only YES or NO.",
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> bool:
    """Ask a yes/no question and return True for YES."""
    answer = chat(prompt, system=system, model=model, temperature=temperature, max_tokens=8)
    return answer.upper().startswith("YES")
