"""Small demo tools so the agent produces visible, distinct spans in traces."""

from __future__ import annotations

import ast
import json
import operator
import re

from langchain_core.tools import tool
from opentelemetry import trace
from opentelemetry.trace import SpanKind

_tracer = trace.get_tracer("galileo_demo.retriever")

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'."""
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


# A tiny built-in knowledge base. Each entry is a retrievable "chunk". The agent
# grounds its answers in these passages, which lets RAG-quality evaluators
# (context adherence, completeness, chunk attribution/utilization) score the run.
_KNOWLEDGE_BASE = [
    {
        "id": "kb-otel",
        "title": "OpenTelemetry",
        "source": "docs/observability.md",
        "content": (
            "OpenTelemetry (OTel) is a vendor-neutral, open-source framework for "
            "generating, collecting, and exporting telemetry data: traces, metrics, "
            "and logs. Applications emit spans over OTLP to a collector, which routes "
            "them to any observability backend."
        ),
    },
    {
        "id": "kb-galileo",
        "title": "Galileo",
        "source": "docs/galileo.md",
        "content": (
            "Galileo is a GenAI observability and evaluation platform. It ingests "
            "traces, including over OpenTelemetry/OTLP, and runs automated quality "
            "metrics such as context adherence, completeness, and chunk attribution "
            "on LLM and agent responses."
        ),
    },
    {
        "id": "kb-nim",
        "title": "NVIDIA NIM",
        "source": "docs/providers.md",
        "content": (
            "NVIDIA NIM provides OpenAI-compatible inference microservices for "
            "optimized model serving. In this demo it serves the Kimi K2 model through "
            "an OpenAI-compatible API endpoint."
        ),
    },
    {
        "id": "kb-ollama",
        "title": "Ollama",
        "source": "docs/providers.md",
        "content": (
            "Ollama runs small open-weight models locally and exposes an "
            "OpenAI-compatible API on port 11434. The demo uses it to run a tiny model "
            "with no external network calls."
        ),
    },
    {
        "id": "kb-arch",
        "title": "Demo architecture",
        "source": "docs/architecture.md",
        "content": (
            "This demo is a vendor-agnostic AI agent chat app. A FastAPI backend runs a "
            "LangGraph ReAct agent that calls an OpenAI-compatible model. The agent is "
            "instrumented with OpenTelemetry, and spans flow over OTLP to a collector "
            "and on to Galileo and other backends."
        ),
    },
    {
        "id": "kb-eval",
        "title": "RAG evaluation metrics",
        "source": "docs/evaluation.md",
        "content": (
            "Context adherence measures whether a response is grounded in retrieved "
            "context. Chunk attribution and chunk utilization measure which retrieved "
            "chunks influenced the answer. Completeness measures how fully the answer "
            "covers the question. These are RAG metrics and require a retrieval step."
        ),
    },
    {
        "id": "kb-vendor-neutral",
        "title": "Vendor-neutral observability",
        "source": "docs/observability.md",
        "content": (
            "Because the app emits standard OpenTelemetry gen_ai spans, switching "
            "observability backends is a collector configuration change, not a code "
            "change. The same trace can fan out to Galileo, Splunk, Phoenix, or Jaeger "
            "at once."
        ),
    },
]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# Common words carry no retrieval signal; matching on them pulls in random
# passages for unrelated prompts (e.g. a cold-email request). Drop them so only
# meaningful query terms count.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "for", "and", "or", "but", "with", "as", "at", "by", "from",
    "this", "that", "these", "those", "it", "its", "i", "you", "he", "she", "we",
    "they", "me", "my", "your", "our", "their", "what", "which", "who", "whom",
    "whose", "when", "where", "why", "how", "do", "does", "did", "can", "could",
    "should", "would", "will", "shall", "may", "might", "must", "have", "has", "had",
    "about", "into", "than", "then", "so", "if", "not", "no", "yes", "please",
    "tell", "write", "give", "explain", "describe", "call", "called", "new", "one",
    "short", "sentence", "relate", "related", "work", "works", "use", "used", "using",
}

# Minimum fraction of meaningful query terms that must appear in a passage for it
# to count as relevant. Keeps unrelated prompts from triggering retrieval at all.
_MIN_SCORE = 0.25


def _retrieve(query: str, k: int = 3) -> list[tuple[dict, float]]:
    """Score the knowledge base against the query by meaningful term overlap."""
    q = _tokenize(query) - _STOPWORDS
    if not q:
        return []
    scored: list[tuple[dict, float]] = []
    for doc in _KNOWLEDGE_BASE:
        terms = _tokenize(f"{doc['title']} {doc['content']}")
        overlap = len(q & terms)
        score = overlap / len(q)
        if score >= _MIN_SCORE:
            scored.append((doc, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]


def retrieve_context(query: str, k: int = 3) -> tuple[str, list[tuple[dict, float]]]:
    """Retrieve relevant knowledge-base passages for a query.

    Runs as a deterministic pre-step (classic retrieve-then-generate), so the
    grounded answer - and the retriever span the RAG evaluators read - is produced
    on every knowledge question regardless of how reliably the model decides to call
    tools. Emits an OpenTelemetry retriever span using open conventions (OTel db.* +
    OpenInference retrieval.documents.*) that any RAG-scoring backend can read.
    Returns ("", []) when nothing matches, so non-knowledge turns stay non-RAG.
    """
    hits = _retrieve(query, k)
    if not hits:
        return "", []

    with _tracer.start_as_current_span(
        "retrieve knowledge_base", kind=SpanKind.CLIENT
    ) as span:
        span.set_attribute("openinference.span.kind", "RETRIEVER")
        span.set_attribute("db.system", "in_memory")
        span.set_attribute("db.operation", "query")
        span.set_attribute("input.value", query)
        span.set_attribute("retrieval.documents.count", len(hits))
        for i, (doc, score) in enumerate(hits):
            prefix = f"retrieval.documents.{i}.document."
            span.set_attribute(prefix + "id", doc["id"])
            span.set_attribute(prefix + "content", doc["content"])
            span.set_attribute(prefix + "score", round(float(score), 4))
            span.set_attribute(
                prefix + "metadata",
                json.dumps({"title": doc["title"], "source": doc["source"]}),
            )

    context = "\n\n".join(
        f"[{doc['id']}] {doc['title']}: {doc['content']}" for doc, _ in hits
    )
    return context, hits


TOOLS = [calculator]
