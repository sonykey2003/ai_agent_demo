"""Vendor-neutral OpenTelemetry setup.

The app only knows how to speak OTLP to a local Collector. The Collector decides
which observability backend(s) receive the data (Galileo, Splunk O11y, Phoenix,
Langfuse, Jaeger, ...). Swapping vendors never touches application code.

GenAI/LLM spans are produced by an instrumentation library if one is installed;
the imports are best-effort so the app still runs in a bare environment.
"""

from __future__ import annotations

import contextvars
import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ..config import get_settings

log = logging.getLogger(__name__)
_initialized = False

# Carries the real GenAI provider/system for the current request. Every provider
# is reached through the same langchain_openai.ChatOpenAI client, so the
# instrumentor's class-name vendor detection always reports "openai". We set this
# per request and let the override below report the actually-selected provider.
active_genai_system: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_genai_system", default=None
)


def set_active_genai_system(system: str | None) -> None:
    """Record the provider/system to stamp on the current request's LLM spans."""
    active_genai_system.set(system)



def setup_telemetry(app=None) -> None:
    """Initialise tracing once and (optionally) instrument a FastAPI app."""
    global _initialized
    if _initialized:
        return

    settings = get_settings()
    if not settings.telemetry_enabled:
        log.info("Telemetry disabled (TELEMETRY_ENABLED=false); skipping OTel setup.")
        _initialized = True
        return

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "deployment.environment": settings.deployment_environment,
        }
    )
    provider = TracerProvider(resource=resource)
    endpoint = settings.otel_exporter_otlp_endpoint.rstrip("/") + "/v1/traces"
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    )
    trace.set_tracer_provider(provider)

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
