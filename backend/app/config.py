"""Application settings and the swappable LLM provider registry.

All three providers (OpenAI, NVIDIA NIM, local Ollama) speak the OpenAI-compatible
API, so a single client with a configurable ``base_url`` / ``model`` / ``api_key``
covers every backend. Switching is a config change, never a code change.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseModel):
    """A single OpenAI-compatible LLM endpoint."""

    name: str
    label: str
    base_url: str
    api_key: str
    model: str
    # Short provider name. Used as the LLM span name (e.g. "NVIDIA NIM") and the
    # gen_ai.provider.name attribute, so traces show the real provider even though
    # every backend is reached through the same OpenAI-compatible client class.
    display: str
    genai_system: str
    # Sampling caps. frequency_penalty > 0 tames repetition on a tiny local model
    # but degrades tool-calling / output quality on capable hosted models, so it is
    # set per provider (0 for OpenAI / NIM).
    max_tokens: int
    frequency_penalty: float


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Which provider to use by default: "openai" | "nim" | "local"
    llm_provider: str = "local"
    temperature: float = 0.2
    # Cap output length so a looping/runaway model can't stream forever.
    max_tokens: int = 1024
    # Small penalty (0.0-2.0) to discourage repetition loops; 0 disables it.
    frequency_penalty: float = 0.3

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # --- NVIDIA NIM (hosted catalog or self-hosted microservice) ---
    nvidia_api_key: str = ""
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_model: str = "deepseek-ai/deepseek-r1"

    # --- Local tiny model via Ollama (OpenAI-compatible endpoint) ---
    ollama_base_url: str = "http://localhost:11434/v1"
    local_model: str = "qwen2.5:0.5b"

    # --- Vendor-neutral OpenTelemetry export ---
    telemetry_enabled: bool = True
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    otel_service_name: str = "agent-chat-demo"
    deployment_environment: str = "demo"

    def providers(self) -> dict[str, ProviderConfig]:
        return {
            "openai": ProviderConfig(
                name="openai",
                label=f"OpenAI · {self.openai_model}",
                base_url=self.openai_base_url,
                api_key=self.openai_api_key or "missing",
                model=self.openai_model,
                display="OpenAI",
                genai_system="openai",
                max_tokens=self.max_tokens,
                frequency_penalty=0.0,
            ),
            "nim": ProviderConfig(
                name="nim",
                label=f"NVIDIA NIM · {self.nim_model}",
                base_url=self.nim_base_url,
                api_key=self.nvidia_api_key or "missing",
                model=self.nim_model,
                display="NVIDIA NIM",
                genai_system="nvidia_nim",
                max_tokens=self.max_tokens,
                frequency_penalty=0.0,
            ),
            "local": ProviderConfig(
                name="local",
                label=f"Local Ollama · {self.local_model}",
                base_url=self.ollama_base_url,
                api_key="ollama",  # Ollama ignores the key but the client requires one.
                model=self.local_model,
                display="Ollama",
                genai_system="ollama",
                max_tokens=self.max_tokens,
                frequency_penalty=self.frequency_penalty,
            ),
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
