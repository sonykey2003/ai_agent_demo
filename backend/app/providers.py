"""Factory that builds an OpenAI-compatible chat model for any provider.

Because OpenAI, NVIDIA NIM, and Ollama all expose the same wire protocol, the only
thing that changes between them is ``base_url`` / ``model`` / ``api_key``.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import get_settings


def _uses_max_completion_tokens(model: str) -> bool:
    """GPT-5 and o-series reject `max_tokens`; they require `max_completion_tokens`."""
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


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
    kwargs = dict(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        model=cfg.model,
        # Names the LLM span after the real provider (e.g. "NVIDIA NIM.chat")
        # instead of the shared client class ("ChatOpenAI.chat").
        name=cfg.display,
        temperature=settings.temperature if temperature is None else temperature,
        frequency_penalty=cfg.frequency_penalty,
        streaming=True,
        stream_usage=settings.llm_stream_usage,
    )
    if _uses_max_completion_tokens(cfg.model):
        # langchain-openai has no max_completion_tokens field, so pass it through.
        kwargs["model_kwargs"] = {"max_completion_tokens": cfg.max_tokens}
    else:
        kwargs["max_tokens"] = cfg.max_tokens
    return ChatOpenAI(**kwargs)
