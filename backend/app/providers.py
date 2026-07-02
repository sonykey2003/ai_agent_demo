"""Factory that builds an OpenAI-compatible chat model for any provider.

Because OpenAI, NVIDIA NIM, and Ollama all expose the same wire protocol, the only
thing that changes between them is ``base_url`` / ``model`` / ``api_key``.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import get_settings


def make_chat_model(
    provider: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    settings = get_settings()
    registry = settings.providers()
    key = provider or settings.llm_provider
    if key not in registry:
        raise ValueError(
            f"Unknown provider '{key}'. Available: {', '.join(registry)}"
        )
    cfg = registry[key]
    return ChatOpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        # Names the LLM span after the real provider (e.g. "NVIDIA NIM.chat")
        # instead of the shared client class ("ChatOpenAI.chat").
        name=cfg.display,
        temperature=settings.temperature if temperature is None else temperature,
        max_tokens=cfg.max_tokens,
        frequency_penalty=cfg.frequency_penalty,
        streaming=True,
    )
