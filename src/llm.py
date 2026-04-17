"""Thin OpenAI wrapper — swap this module to change the LLM backend."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.config import LLM_MAX_TOKENS, LLM_TEMPERATURE, OPENAI_MODEL

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def chat(
    prompt: str,
    *,
    system: str = "You are a helpful assistant.",
    model: str = OPENAI_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    """Send a single-turn chat completion and return the assistant reply."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def yes_no(
    prompt: str,
    *,
    system: str = "You are a helpful assistant. Answer only YES or NO.",
    model: str = OPENAI_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> bool:
    """Ask a yes/no question and return True for YES."""
    answer = chat(prompt, system=system, model=model, temperature=temperature, max_tokens=8)
    return answer.upper().startswith("YES")
