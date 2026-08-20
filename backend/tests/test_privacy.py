from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from app.agent.guardrails import DefaultGuardrail
from app.privacy import redact_pii
from app.telemetry.redacting_exporter import RedactingSpanExporter


class CapturingExporter(SpanExporter):
    def __init__(self) -> None:
        self.spans = []
        self.shutdown_called = False

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.shutdown_called = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def test_shared_redaction_masks_known_pii() -> None:
    raw = "mail jane@example.com phone 415-555-1212 ssn 123-45-6789"
    expected = (
        "mail [REDACTED-EMAIL] phone [REDACTED-PHONE] ssn [REDACTED-SSN]"
    )

    assert redact_pii(raw) == expected
    assert DefaultGuardrail().check_output(raw).redacted_text == expected


def test_exporter_redacts_span_and_event_attributes_without_mutating_source() -> None:
    delegate = CapturingExporter()
    exporter = RedactingSpanExporter(delegate)
    original = ReadableSpan(
        name="llm",
        attributes={
            "gen_ai.input.messages": '"email":"jane@example.com"',
            "values": ("415-555-1212", 7),
        },
        events=(Event("completion", {"output.value": "123-45-6789"}),),
    )

    assert exporter.export((original,)) is SpanExportResult.SUCCESS
    exported = delegate.spans[0]
    assert exported.attributes["gen_ai.input.messages"] == (
        '"email":"[REDACTED-EMAIL]"'
    )
    assert exported.attributes["values"] == ("[REDACTED-PHONE]", 7)
    assert exported.events[0].attributes["output.value"] == "[REDACTED-SSN]"
    assert original.attributes["gen_ai.input.messages"] == '"email":"jane@example.com"'

    assert exporter.force_flush()
    exporter.shutdown()
    assert delegate.shutdown_called