"""Shared Ollama embeddings and PostgreSQL/pgvector store construction."""

from __future__ import annotations

from functools import lru_cache

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from ..config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    settings = get_settings()
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_embedding_base_url,
    )


@lru_cache(maxsize=8)
def get_vector_store(collection: str | None = None) -> PGVector:
    settings = get_settings()
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection or settings.rag_collection,
        connection=settings.postgres_url,
        use_jsonb=True,
    )