import asyncio
import json
import uuid
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

def test_trace_ids_survive_galileo_uuidv4_normalization() -> None:
    generator = otel.UuidV4TraceIdGenerator()
    for _ in range(500):
        trace_id = generator.generate_trace_id()
        # Galileo stores uuid.UUID(int=trace_id, version=4); it must be a no-op.
        assert uuid.UUID(int=trace_id, version=4).int == trace_id


def test_native_turn_sends_full_apm_only_trace_with_shared_id(monkeypatch) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(id_generator=otel.UuidV4TraceIdGenerator())
    provider.add_span_processor(otel._DomainStampingSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        otel.trace, "get_tracer", lambda name, *a, **k: provider.get_tracer(name)
    )

    class FakeLogger:
        mode = "batch"
        metadata = None
        trace = None

        def start_trace(self, *, input, name, metadata=None):
            self.metadata = metadata
            self.trace = SimpleNamespace(id=uuid.uuid4())
            return self.trace

        def conclude(self, output):
            pass

        def flush(self):
            pass

    class FakeAgent:
        async def astream_events(self, payload, *, config, version):
            # Stands in for the retriever/LLM spans: they must reach APM now.
            with otel.trace.get_tracer("inner").start_as_current_span("inner"):
                pass
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": SimpleNamespace(content="hi")},
            }

    async def passthrough(text):
        return text

    async def no_output_control(answer, req):
        return answer, None

    fake_logger = FakeLogger()
    monkeypatch.setattr(main, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(main, "agent_control_active", lambda: True)
    monkeypatch.setattr(main, "otel_sink_active", lambda: False)
    monkeypatch.setattr(main, "get_ac_logger", lambda domain: fake_logger)
    monkeypatch.setattr(main, "evaluate_user_input", passthrough)
    monkeypatch.setattr(main, "_resolve_output_control", no_output_control)
    monkeypatch.setattr(main, "build_agent", lambda **kwargs: FakeAgent())

    asyncio.run(
        _events(ChatRequest(message="hello", provider="local", domain="bank"))
    )

    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert set(spans) == {"POST /chat", "invoke_agent Agent Chat", "inner"}
    assert all(s.attributes[otel.APM_ONLY_ATTR] is True for s in spans.values())
    turn = spans["invoke_agent Agent Chat"]
    assert spans["inner"].context.trace_id == turn.context.trace_id
    assert fake_logger.trace.id.int == turn.context.trace_id
    assert turn.attributes["galileo.trace_id"] == str(fake_logger.trace.id)
    assert turn.attributes["output.value"] == "hi"
    assert fake_logger.metadata == {
        "otel_trace_id": format(turn.context.trace_id, "032x")
    }
    provider.shutdown()
