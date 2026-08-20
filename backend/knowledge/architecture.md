# Demo architecture

The FastAPI backend runs a LangGraph ReAct agent and streams responses to the web
interface with server-sent events. Before the first model call, a deterministic
pre-model hook searches PostgreSQL with pgvector and adds relevant passages to the
model input. OpenTelemetry spans flow through a local Collector to Galileo or any
other configured OTLP backend.