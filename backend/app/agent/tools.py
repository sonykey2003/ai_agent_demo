"""Small demo tools so the agent produces visible, distinct spans in traces."""

from __future__ import annotations

import ast
import contextvars
import json
import logging
import operator
import re
from collections.abc import Iterator
from contextlib import contextmanager

from langchain_core.tools import tool
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ..config import get_settings
from ..rag.vector_store import get_vector_store
from .agent_control import ControlSteerError, ControlViolationError, control
from .bank_db import query_customers

_tracer = trace.get_tracer("galileo_demo.retriever")
log = logging.getLogger(__name__)

# Carries the pgvector collection for the current request so retrieval (which
# runs inside the agent's pre-model hook) grounds on the UI-selected domain.
active_rag_collection: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_rag_collection", default=None
)


@contextmanager
def rag_collection_scope(collection: str | None) -> Iterator[None]:
    """Scope the active retrieval collection to one request."""
    token = active_rag_collection.set(collection)
    try:
        yield
    finally:
        active_rag_collection.reset(token)

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


# Bank-only DB tool. The query runs as an Agent Control step ("query_customer_db")
# so a control can govern it (e.g. block destructive SQL) exactly like the
# getting-started-labs guardrails demo; it's a no-op passthrough when no control
# is bound to that step. The .name/.tool_name marks it as a TOOL step so Agent
# Control exposes the args as a dict (input.sql), not a flattened string.
query_customers.name = "query_customer_db"
query_customers.tool_name = "query_customer_db"
_query_customers_controlled = control(step_name="query_customer_db")(query_customers)

_ROW_CAP_RE = re.compile(r"\b(\d+)\b")
_HAS_LIMIT_RE = re.compile(r"\blimit\b", re.IGNORECASE)


def _revised_sql(sql: str, guidance: str) -> str | None:
    """Apply a row-cap steer to `sql` so the retry is one deterministic call.

    Returning only prose guidance makes the model resubmit the identical SQL and
    trip the control a second time; handing it the exact statement to re-send
    keeps the turn at one steered call plus one allowed call.
    """
    statement = sql.strip().rstrip(";").strip()
    if not statement.lower().startswith("select") or _HAS_LIMIT_RE.search(statement):
        return None
    cap = _ROW_CAP_RE.search(guidance)
    if not cap:
        return None
    return f"{statement} LIMIT {cap.group(1)}"


@tool
def query_customer_db(sql: str) -> str:
    """Run a SQL statement against the bank database (policy-governed).

    Tables:
      customers(id TEXT, name TEXT, email TEXT, account_type TEXT, balance REAL)
        - account_type is one of 'Savings', 'Checking', 'Premier'.
      transactions(id TEXT, account_id TEXT, txn_date TEXT, merchant TEXT,
                   category TEXT, amount REAL)
        - account_id joins customers.id; ~42 rows of card/account activity.
    Reads use SELECT, e.g. SELECT merchant, amount FROM transactions WHERE category = 'Travel'.
    Modification requests (DELETE / UPDATE) are passed through as written; a separate
    query-layer policy decides whether they run. Returns JSON: {"success", "row_count", "data": [...]}.
    If the policy steers the query the result is {"success": false, "revise_query", "rejected_sql",
    "retry_with"} — call this tool once more with "retry_with" verbatim; never resend "rejected_sql".
    """
    # Keyword so Agent Control resolves the value at its "input.sql" path.
    try:
        return _query_customers_controlled(sql=sql)
    except ControlSteerError as exc:
        # Steer: return the guidance so the model revises the SQL and retries.
        guidance = str(getattr(exc, "steering_context", "") or exc)
        payload = {
            "success": False,
            "revise_query": guidance,
            "rejected_sql": sql,
            "data": [],
        }
        retry_with = _revised_sql(sql, guidance)
        if retry_with:
            payload["retry_with"] = retry_with
        return json.dumps(payload)
    except ControlViolationError as exc:
        # Deny: report the block so the model tells the user (no retry).
        return json.dumps(
            {"success": False, "blocked_by_policy": str(exc), "data": []}
        )


def _retrieve(query: str, k: int, collection: str | None) -> list[tuple[dict, float]]:
    """Return pgvector similarity hits in the app's stable document shape."""
    settings = get_settings()
    matches = get_vector_store(collection).similarity_search_with_relevance_scores(
        query,
        k=k,
        score_threshold=settings.rag_score_threshold,
    )
    return [
        (
            {
                "id": document.metadata.get("citation_id")
                or document.metadata.get("chunk_id")
                or document.id
                or "unknown",
                "title": document.metadata.get("title")
                or document.metadata.get("source", "Knowledge document"),
                "source": document.metadata.get("source", "unknown"),
                "content": document.page_content,
            },
            float(score),
        )
        for document, score in matches
    ]


def retrieve_context(
    query: str, k: int | None = None, collection: str | None = None
) -> tuple[str, list[tuple[dict, float]]]:
    """Retrieve relevant knowledge-base passages for a query.

    Runs as a deterministic pre-step (classic retrieve-then-generate), so the
    grounded answer - and the retriever span the RAG evaluators read - is produced
    on every knowledge question regardless of how reliably the model decides to call
    tools. Emits an OpenTelemetry retriever span using open conventions (OTel db.* +
    OpenInference retrieval.documents.*) that any RAG-scoring backend can read.
    Returns ("", []) when nothing matches, so non-knowledge turns stay non-RAG.

    The collection is passed explicitly by the caller (bound to the selected domain
    at agent-build time); it falls back to the request-scoped contextvar. If no
    collection is bound, retrieval is skipped - it never silently queries the
    default/platform collection.
    """
    settings = get_settings()
    if not settings.rag_enabled:
        return "", []

    collection = collection or active_rag_collection.get()
    if collection is None:
        log.warning(
            "No RAG collection bound for this request; skipping retrieval "
            "instead of querying the default collection"
        )
        return "", []
    with _tracer.start_as_current_span(
        "retrieve knowledge_base", kind=SpanKind.CLIENT
    ) as span:
        span.set_attribute("openinference.span.kind", "RETRIEVER")
        span.set_attribute("db.system", "postgresql")
        span.set_attribute("db.operation", "query")
        span.set_attribute("db.namespace", collection)
        span.set_attribute("input.value", query)
        try:
            hits = _retrieve(query, k or settings.rag_top_k, collection)
        except Exception as exc:  # noqa: BLE001
            log.warning("RAG retrieval unavailable: %s", exc)
            span.record_exception(exc)
            span.set_attribute("error.type", type(exc).__name__)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            return "", []

        span.set_attribute("retrieval.documents.count", len(hits))
        span.set_attribute("retrieval.top_k", k or settings.rag_top_k)
        span.set_attribute("retrieval.score_threshold", settings.rag_score_threshold)
        span.set_attribute("embedding.model", settings.ollama_embedding_model)
        for i, (doc, score) in enumerate(hits):
            prefix = f"retrieval.documents.{i}.document."
            span.set_attribute(prefix + "id", doc["id"])
            span.set_attribute(prefix + "content", doc["content"])
            span.set_attribute(prefix + "score", round(float(score), 4))
            span.set_attribute(
                prefix + "metadata",
                json.dumps({"title": doc["title"], "source": doc["source"]}),
            )
        # Galileo reads retriever output as a document list and requires each to
        # carry page_content; this populates the chunk display + RAG scoring.
        span.set_attribute(
            "output.value",
            json.dumps(
                [
                    {
                        "page_content": d["content"],
                        "metadata": {
                            "id": d["id"],
                            "title": d["title"],
                            "source": d["source"],
                            "score": round(float(s), 4),
                        },
                    }
                    for d, s in hits
                ]
            ),
        )

    if not hits:
        return "", []

    # Rerank step: currently a passthrough that preserves similarity order. Emits
    # an OpenInference RERANKER span so the pipeline reads retrieve -> rerank ->
    # augment; swap the body for a cross-encoder to make it real.
    with _tracer.start_as_current_span(
        "rerank documents", kind=SpanKind.INTERNAL
    ) as rr:
        rr.set_attribute("openinference.span.kind", "RERANKER")
        rr.set_attribute("reranker.model_name", "relevance-threshold")
        rr.set_attribute("reranker.query", query)
        rr.set_attribute("input.value", query)
        for i, (doc, score) in enumerate(hits):
            p = f"reranker.input_documents.{i}.document."
            rr.set_attribute(p + "id", doc["id"])
            rr.set_attribute(p + "content", doc["content"])
            rr.set_attribute(p + "score", round(float(score), 4))
        # Precision cut: keep only strongly-relevant candidates so off-topic
        # chunks don't pollute the prompt context. Fall back to the single best
        # hit so a knowledge turn is never left with no context.
        reranked = [
            (d, s) for d, s in hits if s >= settings.rag_rerank_min_score
        ] or hits[:1]
        rr.set_attribute("reranker.top_k", len(reranked))
        for i, (doc, score) in enumerate(reranked):
            p = f"reranker.output_documents.{i}.document."
            rr.set_attribute(p + "id", doc["id"])
            rr.set_attribute(p + "content", doc["content"])
            rr.set_attribute(p + "score", round(float(score), 4))
        rr.set_attribute(
            "output.value",
            json.dumps(
                [
                    {
                        "page_content": d["content"],
                        "metadata": {"id": d["id"], "score": round(float(s), 4)},
                    }
                    for d, s in reranked
                ]
            ),
        )

    # Discrete augmentation step (documents -> prompt context), emitted as a tool
    # span so it appears as its own step for RAG observability (parity with the
    # golden demo's "prompt-augmentation" tool span).
    with _tracer.start_as_current_span(
        "prompt-augmentation", kind=SpanKind.INTERNAL
    ) as aug:
        aug.set_attribute("gen_ai.operation.name", "execute_tool")
        aug.set_attribute("gen_ai.tool.name", "prompt-augmentation")
        context = "\n\n".join(
            f"[{doc['id']}] {doc['title']}: {doc['content']}" for doc, _ in reranked
        )
        arguments = json.dumps(
            {
                "task": "format retrieved documents into prompt context",
                "document_ids": [doc["id"] for doc, _ in reranked],
            }
        )
        # Galileo reads a tool span's I/O from gen_ai.tool.call.arguments/result;
        # input.value/output.value are kept as fallbacks.
        aug.set_attribute("gen_ai.tool.call.arguments", arguments)
        aug.set_attribute("input.value", arguments)
        aug.set_attribute("gen_ai.tool.call.result", context)
        aug.set_attribute("output.value", context)
        aug.set_attribute("num_docs", len(reranked))
        aug.set_attribute("context.chars", len(context))
    return context, reranked


TOOLS = [calculator]
_BANK_TOOLS = [calculator, query_customer_db]


def tools_for_domain(domain: str | None) -> list:
    """Bank gets the customer-DB tool; every other domain stays calculator-only."""
    return _BANK_TOOLS if domain == "bank" else TOOLS
