"""Vendor-neutral OpenTelemetry setup.

The app only knows how to speak OTLP to a local Collector. The Collector decides
which observability backend(s) receive the data (Galileo, Splunk O11y, Phoenix,
Langfuse, Jaeger, ...). Swapping vendors never touches application code.

GenAI/LLM spans are produced by an instrumentation library if one is installed;
the imports are best-effort so the app still runs in a bare environment.
"""

from __future__ import annotations

import contextvars
import json
import logging
from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span as SdkSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import (
    ALWAYS_ON,
    Decision,
    ParentBased,
    Sampler,
    SamplingResult,
)
from opentelemetry.trace import Span, SpanKind

from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

from ..config import get_settings

log = logging.getLogger(__name__)
_initialized = False
_provider: TracerProvider | None = None
_logger_provider: LoggerProvider | None = None
_shutdown = False

# Carries the real GenAI provider/system for the current request. Every provider
# is reached through the same langchain_openai.ChatOpenAI client, so the
# instrumentor's class-name vendor detection always reports "openai". We set this
# per request and let the override below report the actually-selected provider.
active_genai_system: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_genai_system", default=None
)


@contextmanager
def active_genai_system_scope(system: str | None) -> Generator[None]:
    """Scope the provider override to one request and restore it afterwards."""
    token = active_genai_system.set(system)
    try:
        yield
    finally:
        active_genai_system.reset(token)


# Carries the UI-selected domain for the current request so every span in the
# turn (not just the root) can be tagged with it. The Collector routes whole
# traces to a per-domain Galileo log stream by this attribute.
active_domain: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_domain", default=None
)


@contextmanager
def active_domain_scope(name: str | None) -> Generator[None]:
    """Scope the active domain to one request and restore it afterwards."""
    token = active_domain.set(name)
    try:
        yield
    finally:
        active_domain.reset(token)


# Set for turns the native Galileo logger owns end to end. OTel's
# suppress-instrumentation context key is only honoured by OTel-native
# instrumentors, so OpenLLMetry's LangChain handler and this app's own manual
# spans still emit and surface as a second, half-empty trace next to the native
# one. Dropping at the sampler stops every span regardless of who created it.
otel_muted: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "otel_muted", default=False
)


@contextmanager
def mute_otel_scope() -> Generator[None]:
    """Drop every OTel span created within this scope."""
    token = otel_muted.set(True)
    try:
        yield
    finally:
        otel_muted.reset(token)


class _MutableSampler(Sampler):
    """Delegating sampler that drops spans while ``otel_muted`` is set."""

    def __init__(self, delegate: Sampler) -> None:
        self._delegate = delegate

    def should_sample(self, *args, **kwargs) -> SamplingResult:  # noqa: D102
        if otel_muted.get():
            return SamplingResult(Decision.DROP)
        return self._delegate.should_sample(*args, **kwargs)

    def get_description(self) -> str:  # noqa: D102
        return f"MutableSampler({self._delegate.get_description()})"


class _DomainStampingSpanProcessor(SpanProcessor):
    """Stamp the routing attribute on every span at start.

    ``domain.name`` is set on the root turn span directly, but child spans
    (LLM, tool, retriever, LangGraph) are created by the instrumentor and would
    otherwise lack it. Stamping at ``on_start`` from the per-request contextvar
    puts it on all spans, so the Collector can route the whole trace to one
    per-domain Galileo log stream.
    """

    def on_start(self, span: SdkSpan, parent_context: Context | None = None) -> None:
        domain = active_domain.get()
        if domain:
            span.set_attribute("domain.name", domain)

    def on_end(self, span: SdkSpan) -> None:  # noqa: D102
        pass

    def shutdown(self) -> None:  # noqa: D102
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # noqa: D102
        return True


def _genai_message(role: str, content: str, finish_reason: str | None = None) -> dict:
    msg: dict = {"role": role, "parts": [{"type": "text", "content": content}]}
    if finish_reason is not None:
        msg["finish_reason"] = finish_reason
    return msg


def set_turn_input(span: Span, text: str) -> None:
    """Record the real user input on the turn root span."""
    span.set_attribute("input.value", text)
    span.set_attribute(
        "gen_ai.input.messages", json.dumps([_genai_message("user", text)])
    )


def set_turn_output(span: Span, text: str) -> None:
    """Record the real assistant output on the turn root span.

    The LangChain instrumentor writes gen_ai.output.messages from the aggregated
    LLM result, which is empty for streamed (astream_events) turns; Galileo shows
    those empty messages over input.value/output.value. Writing the captured text
    here (after the instrumentor's callbacks) restores the real content.
    """
    span.set_attribute("output.value", text)
    span.set_attribute(
        "gen_ai.output.messages",
        json.dumps([_genai_message("assistant", text, finish_reason="stop")]),
    )


@contextmanager
def start_chat_span(
    *,
    conversation_id: str,
    provider_name: str,
    model: str,
    input_text: str,
) -> Generator[Span]:
    """Create the valid GenAI root span that owns one complete streamed turn."""
    tracer = trace.get_tracer("agent_chat.turn")
    with tracer.start_as_current_span(
        "invoke_agent Agent Chat", kind=SpanKind.INTERNAL
    ) as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.provider.name", provider_name)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.agent.name", "Agent Chat")
        span.set_attribute("gen_ai.conversation.id", conversation_id)
        span.set_attribute("session.id", conversation_id)
        set_turn_input(span, input_text)
        yield span


@contextmanager
def start_server_span(*, method: str, route: str) -> Generator[Span]:
    """Entry-point span for APM; created inside the SSE generator so it wraps the real turn."""
    tracer = trace.get_tracer("agent_chat.http")
    with tracer.start_as_current_span(f"{method} {route}", kind=SpanKind.SERVER) as span:
        span.set_attribute("http.request.method", method)
        span.set_attribute("http.route", route)
        span.set_attribute("url.path", route)
        span.set_attribute("http.response.status_code", 200)
        yield span

@contextmanager
def start_guardrail_span(stage: str) -> Generator[Span]:
    """Create a child span for a generic input or output guardrail check."""
    tracer = trace.get_tracer("agent_chat.guardrails")
    with tracer.start_as_current_span(f"guardrail {stage}") as span:
        span.set_attribute("guardrail.stage", stage)
        yield span


def set_guardrail_result(
    span: Span,
    *,
    allowed: bool,
    reason: str = "",
    redacted: bool = False,
) -> None:
    """Record a generic guardrail decision without classifying it as GenAI work."""
    span.set_attribute("guardrail.allowed", allowed)
    action = "redact" if redacted else "allow" if allowed else "block"
    span.set_attribute("guardrail.action", action)
    span.set_attribute("guardrail.redacted", redacted)
    if reason:
        span.set_attribute("guardrail.reason", reason)

def setup_telemetry(app=None) -> TracerProvider | None:
    """Initialise tracing once and (optionally) instrument a FastAPI app."""
    global _initialized, _provider, _shutdown, _logger_provider
    if _initialized:
        return _provider

    settings = get_settings()
    if not settings.telemetry_enabled:
        log.info("Telemetry disabled (TELEMETRY_ENABLED=false); skipping OTel setup.")
        _initialized = True
        return None

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "deployment.environment": settings.deployment_environment,
        }
    )
    provider = TracerProvider(
        resource=resource, sampler=_MutableSampler(ParentBased(ALWAYS_ON))
    )
    endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/") + "/v1/traces"
    exporter = OTLPSpanExporter(endpoint=endpoint)
    # Runs before the exporter so every span carries domain.name for routing.
    provider.add_span_processor(_DomainStampingSpanProcessor())
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _logger_provider = LoggerProvider(resource=resource)
    _logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(
                endpoint=settings.otel_exporter_otlp_endpoint.rstrip("/") + "/v1/logs"
            )
        )
    )
    set_logger_provider(_logger_provider)
    handler = LoggingHandler(level=logging.INFO, logger_provider=_logger_provider)
    # Exporter failures log to root; re-exporting them would feed the exporter its own errors.
    handler.addFilter(lambda record: not record.name.startswith("opentelemetry"))
    logging.getLogger().addHandler(handler)
    _provider = provider
    _shutdown = False

    # NOTE: FastAPI/HTTP instrumentation is intentionally NOT enabled. The chat
    # endpoint streams via SSE, so the agent runs in the response generator AFTER
    # the HTTP server span has closed — its context is detached, producing a
    # separate, empty "POST /chat" trace alongside the real one. Skipping it keeps
    # exactly one clean GenAI trace per chat. (Re-add FastAPIInstrumentor here if
    # you want HTTP-level spans for a non-streaming backend.)

    # GenAI spans via OpenLLMetry's LangChain/LangGraph instrumentor. It emits the
    # OTEL gen_ai.* semantic conventions (gen_ai.operation.name, gen_ai.provider.name,
    # span names like "invoke_agent" / "execute_tool") that Galileo's OTLP provider
    # requires to classify spans — producing one coherent agent -> LLM -> tool trace.
    # Vendor-neutral: the same gen_ai.* spans work with any OTLP backend.
    try:
        from opentelemetry.instrumentation.langchain import LangchainInstrumentor

        LangchainInstrumentor().instrument(tracer_provider=provider)
    except Exception as exc:  # noqa: BLE001
        log.warning("LangChain (OpenLLMetry) instrumentation unavailable: %s", exc)

    # All providers share langchain_openai.ChatOpenAI (an OpenAI-compatible client),
    # so OpenLLMetry's class-name vendor detection labels every LLM span "openai".
    # Override it to report the provider actually selected for the request (set via
    # set_active_genai_system). Best-effort: if internals change, spans simply keep
    # the default detection.
    try:
        from opentelemetry.instrumentation.langchain import callback_handler as _cb

        _orig_detect = _cb.detect_vendor_from_class

        def _detect_with_override(class_name: str) -> str:
            return active_genai_system.get() or _orig_detect(class_name)

        _cb.detect_vendor_from_class = _detect_with_override
    except Exception as exc:  # noqa: BLE001
        log.warning("Provider-name override unavailable: %s", exc)

    _initialized = True
    log.info("OpenTelemetry initialised; exporting OTLP to %s", endpoint)
    return provider


def force_flush_telemetry(timeout_millis: int = 30000) -> bool:
    """Flush buffered spans when telemetry is active."""
    if _provider is None or _shutdown:
        return True
    return _provider.force_flush(timeout_millis)


def shutdown_telemetry() -> None:
    """Flush and stop the configured provider exactly once."""
    global _shutdown
    if _provider is None or _shutdown:
        return
    force_flush_telemetry()
    _provider.shutdown()
    if _logger_provider is not None:
        _logger_provider.shutdown()
    _shutdown = True
