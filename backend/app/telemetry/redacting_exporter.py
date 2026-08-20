"""Trace exporter decorator that removes known PII before the OTLP hop."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from ..privacy import redact_pii


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def _redact_attributes(attributes) -> dict[str, Any] | None:
    if attributes is None:
        return None
    return {key: _redact_value(value) for key, value in attributes.items()}


def _redact_span(span: ReadableSpan) -> ReadableSpan:
    # Honor the per-request guardrails toggle stamped on every span: when the
    # app's guardrails are off, leave content raw so an external guardrail
    # (e.g. Galileo) can evaluate the real input/output.
    if (span.attributes or {}).get("app.guardrails.enabled") is False:
        return span
    events = tuple(
        Event(
            event.name,
            attributes=_redact_attributes(event.attributes),
            timestamp=event.timestamp,
        )
        for event in span.events
    )
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_redact_attributes(span.attributes),
        events=events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class RedactingSpanExporter(SpanExporter):
    """Sanitize span content before delegating to another exporter."""

    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._exporter.export(tuple(_redact_span(span) for span in spans))

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)
