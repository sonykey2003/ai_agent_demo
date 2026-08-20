# OpenTelemetry

OpenTelemetry, or OTel, is a vendor-neutral open-source framework for generating,
collecting, and exporting traces, metrics, and logs. This application emits spans
over OTLP to an OpenTelemetry Collector. The Collector decides which observability
backends receive those spans, so changing backends does not require changing the
agent's runtime instrumentation.