"""Google Colab AI wrapper — uses Colab's built-in LLM, no API keys needed."""

from __future__ import annotations

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
