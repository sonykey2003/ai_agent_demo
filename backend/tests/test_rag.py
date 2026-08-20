import asyncio
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from app.agent import tools
from app import main
from app.rag import ingest


def test_load_documents_and_stable_chunk_id(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("Vector search uses embeddings.")
    documents = ingest._load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].metadata == {"source": "guide.md", "title": "Guide"}
    assert ingest._chunk_id(documents[0]) == ingest._chunk_id(documents[0])


def test_default_source_dir_contains_bundled_knowledge() -> None:
    assert (ingest.default_source_dir() / "otel.md").is_file()


def test_lifespan_builds_rag_before_startup(monkeypatch) -> None:
    calls = []
    settings = SimpleNamespace(
        rag_enabled=True,
        rag_collection="test_knowledge",
        rag_startup_attempts=1,
        rag_startup_retry_seconds=0,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "setup_telemetry", lambda _app: calls.append("setup"))
    monkeypatch.setattr(main, "shutdown_telemetry", lambda: calls.append("shutdown"))
    monkeypatch.setattr(
        main,
        "ingest_all",
        lambda *, reset: calls.append(("ingest_all", reset)) or {},
    )

    async def run_lifespan() -> None:
        async with main.lifespan(main.app):
            calls.append("running")

    asyncio.run(run_lifespan())

    assert calls == [
        "setup",
        ("ingest_all", True),
        "running",
        "shutdown",
    ]


def test_retrieve_context_uses_pgvector_and_emits_documents(monkeypatch) -> None:
    class FakeStore:
        def similarity_search_with_relevance_scores(self, query, *, k, score_threshold):
            assert query == "How does telemetry export work?"
            assert k == 2
            assert score_threshold == 0.45
            return [
                (
                    Document(
                        page_content="Applications export spans over OTLP.",
                        metadata={
                            "citation_id": "otel-0",
                            "source": "otel.md",
                            "title": "OpenTelemetry",
                        },
                    ),
                    0.91,
                )
            ]

    settings = SimpleNamespace(
        rag_enabled=True,
        rag_collection="test_knowledge",
        rag_score_threshold=0.45,
        rag_top_k=3,
    )
    monkeypatch.setattr(tools, "get_settings", lambda: settings)
    monkeypatch.setattr(tools, "get_vector_store", lambda collection=None: FakeStore())

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        tools, "_tracer", provider.get_tracer("test.rag.retriever")
    )

    context, hits = tools.retrieve_context(
        "How does telemetry export work?", k=2
    )

    assert "[otel-0] OpenTelemetry" in context
    assert hits[0][1] == 0.91
    span = exporter.get_finished_spans()[0]
    assert span.attributes["db.system"] == "postgresql"
    assert span.attributes["db.namespace"] == "test_knowledge"
    assert span.attributes["retrieval.documents.0.document.id"] == "otel-0"
    assert (
        span.attributes["retrieval.documents.0.document.content"]
        == "Applications export spans over OTLP."
    )
    provider.shutdown()