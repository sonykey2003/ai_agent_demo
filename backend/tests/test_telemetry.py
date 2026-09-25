import asyncio
import json
from types import SimpleNamespace

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from app import main
from app.config import Settings
from app.main import ChatRequest
from app.telemetry import otel


def test_provider_scope_restores_previous_value() -> None:
    token = otel.active_genai_system.set("before")
    try:
        with otel.active_genai_system_scope("ollama"):
            assert otel.active_genai_system.get() == "ollama"
        assert otel.active_genai_system.get() == "before"
    finally:
        otel.active_genai_system.reset(token)


def test_chat_and_guardrail_span_attributes(monkeypatch) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        otel.trace, "get_tracer", lambda name: provider.get_tracer(name)
    )

    with otel.start_chat_span(
        conversation_id="conversation-1",
        provider_name="ollama",
        model="qwen2.5:0.5b",
        input_text="hello",
    ) as root:
        with otel.start_guardrail_span("input") as guardrail:
            otel.set_guardrail_result(guardrail, allowed=False, reason="blocked")
        root.set_attribute("output.value", "blocked")

    spans = {span.name: span for span in exporter.get_finished_spans()}
    root = spans["invoke_agent Agent Chat"]
    guardrail = spans["guardrail input"]
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert root.attributes["gen_ai.conversation.id"] == "conversation-1"
    assert root.attributes["session.id"] == "conversation-1"
    assert guardrail.attributes["guardrail.action"] == "block"
    assert guardrail.parent.span_id == root.context.span_id
    provider.shutdown()


def test_chat_request_generates_distinct_conversation_ids() -> None:
    assert ChatRequest(message="one").conversation_id != ChatRequest(
        message="two"
    ).conversation_id


def test_openrouter_is_available_as_a_provider() -> None:
    openrouter = Settings(_env_file=None).providers()["openrouter"]

    assert openrouter.base_url == "https://openrouter.ai/api/v1"
    assert openrouter.genai_system == "openrouter"


async def _events(request: ChatRequest) -> list[dict]:
    return [
        json.loads(chunk.removeprefix("data: ").strip())
        async for chunk in main._event_stream(request)
    ]


def test_stream_passes_thread_id_and_leaves_output_raw(monkeypatch) -> None:
    captured = {}

    class FakeAgent:
        async def astream_events(self, payload, *, config, version):
            captured["config"] = config
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content="jane@example.com")},
            }

    monkeypatch.setattr(main, "build_agent", lambda **kwargs: FakeAgent())
    # Pin Agent Control off so the assertion is about the app's own behaviour and
    # not about whichever controls happen to be bound in the Galileo console.
    monkeypatch.setattr(main, "agent_control_active", lambda: False)
    events = asyncio.run(
        _events(
            ChatRequest(
                message="contact",
                provider="local",
                conversation_id="conversation-2",
            )
        )
    )

    assert captured["config"]["configurable"]["thread_id"] == "conversation-2"
    # The app no longer redacts; Galileo Agent Control owns every control decision.
    assert not [e for e in events if e["type"] == "redacted"]
    assert {"type": "token", "text": "jane@example.com"} in events
    assert events[-1] == {"type": "done"}