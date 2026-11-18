"""Factory that builds an OpenAI-compatible chat model for any provider.

Because OpenAI, OpenRouter, and Ollama all expose the same wire protocol, the only
thing that changes between them is ``base_url`` / ``model`` / ``api_key``.
"""

from __future__ import annotations

import logging

import httpx
from langchain_openai import ChatOpenAI

from .config import ProviderConfig, get_settings

log = logging.getLogger(__name__)

# Only successful lookups are cached, so a capability check made while Ollama is
# still starting is retried on the next request.
_VISION_CACHE: dict[tuple[str, str], bool] = {}


def _ollama_native_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def _ollama_has_vision(base_url: str, model: str) -> bool:
    key = (base_url, model)
    if key in _VISION_CACHE:
        return _VISION_CACHE[key]
    try:
        res = httpx.post(
            _ollama_native_url(base_url) + "/api/show",
            json={"model": model},
            timeout=3.0,
        )
        res.raise_for_status()
        capabilities = res.json().get("capabilities") or []
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read Ollama capabilities for %s: %s", model, exc)
        return False
    _VISION_CACHE[key] = "vision" in capabilities
    return _VISION_CACHE[key]


def provider_supports_vision(cfg: ProviderConfig) -> bool:
    """Whether the provider's model accepts image input."""
    if cfg.supports_vision is not None:
        return cfg.supports_vision
    if cfg.genai_system == "ollama":
        return _ollama_has_vision(cfg.base_url, cfg.model)
    return False


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
        # Names the LLM span after the real provider (e.g. "OpenRouter.chat")
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
