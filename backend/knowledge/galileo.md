# Galileo

Galileo is a GenAI observability and evaluation platform. It accepts traces over
OpenTelemetry OTLP and evaluates agent and LLM behavior. Its RAG quality metrics
include context adherence, completeness, chunk attribution, and chunk utilization.
Those metrics rely on retriever spans containing the query and retrieved document
content.