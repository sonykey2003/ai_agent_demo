"""FastAPI entrypoint: streaming chat over an agent, plus provider listing.

The chat endpoint streams Server-Sent Events so the UI can render tokens, tool
calls, and guardrail notices as they happen. The static chat UI is served from
the same app to keep the demo to a single container.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry import context as otel_context
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field

from .agent.graph import build_agent
from .agent.guardrails import get_guardrail
from .agent.agent_control import (
    ControlSteerError,
    ControlViolationError,
    bind_otel_trace_context,
    evaluate_user_input,
    get_ac_logger,
    is_active as agent_control_active,
    otel_sink_active,
    setup_agent_control,
    unbind_otel_trace_context,
)
from .agent.tools import rag_collection_scope
from .config import get_settings
from .domains import DEFAULT_DOMAIN, get_domains
from .rag.ingest import ingest_all
from .telemetry.otel import (
    active_domain_scope,
    active_genai_system_scope,
    active_guardrails_scope,
    set_guardrail_result,
    set_turn_input,
    set_turn_output,
    setup_telemetry,
    shutdown_telemetry,
    start_chat_span,
    start_guardrail_span,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


async def _initialize_rag() -> None:
    settings = get_settings()
    if not settings.rag_enabled:
        return

    for attempt in range(1, settings.rag_startup_attempts + 1):
        try:
            counts = await asyncio.to_thread(ingest_all, reset=True)
            for name, count in counts.items():
                log.info(
                    "RAG ready: indexed %d chunks into %s (domain %s)",
                    count,
                    get_domains()[name].collection,
                    name,
                )
            return
        except Exception:  # noqa: BLE001
            if attempt == settings.rag_startup_attempts:
                log.exception(
                    "RAG startup failed after %d attempts", attempt
                )
                raise
            log.warning(
                "RAG startup attempt %d/%d failed; retrying in %.1fs",
                attempt,
                settings.rag_startup_attempts,
                settings.rag_startup_retry_seconds,
            )
            await asyncio.sleep(settings.rag_startup_retry_seconds)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_telemetry(_app)
    setup_agent_control()
    try:
        await _initialize_rag()
        yield
    finally:
        shutdown_telemetry()


app = FastAPI(title="Vendor-Agnostic AI Agent Chat Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_guardrail = get_guardrail()
_STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str
    provider: str | None = None
    domain: str | None = None
    temperature: float | None = None
    # When false, the app's built-in input-block and output-PII-redaction
    # guardrails are skipped so an external guardrail (e.g. Galileo) can be
    # demoed on the raw input/output without interference.
    guardrails_enabled: bool = True
    conversation_id: str = Field(
        default_factory=lambda: str(uuid4()), min_length=1, max_length=128
    )


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _chunk_text(chunk) -> str:
    """Extract text from a streamed chat-model chunk (string or content blocks)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for piece in content:
            if isinstance(piece, dict):
                parts.append(piece.get("text") or piece.get("content") or "")
            else:
                parts.append(str(piece))
        return "".join(parts)
    return str(content or "")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/providers")
def providers() -> dict:
    settings = get_settings()
    return {
        "default": settings.llm_provider,
        "providers": [
            {"id": p.name, "label": p.label, "model": p.model}
            for p in settings.providers().values()
        ],
    }


@app.get("/domains")
def domains() -> dict:
    return {
        "default": DEFAULT_DOMAIN,
        "domains": [
            {"id": d.name, "label": d.label} for d in get_domains().values()
        ],
    }


async def _agent_control_stream(req: ChatRequest, cfg, domain):
    """Agent Control turn traced by the native Galileo logger (reference pattern).

    The logger sets the trace input/output explicitly and enable_agent_control()
    bridges the @control span into the same trace, so input, the control decision,
    and output all render. OTel is suppressed so the logger owns one clean trace.
    """
    logger = get_ac_logger()
    yield _sse(
        {"type": "meta", "domain": domain.name, "collection": domain.collection}
    )

    logger.start_trace(input=req.message, name="Agent Chat")
    concluded = False
    try:
        # Enforcement is the @control decorator: a deny/steer raises here.
        try:
            await evaluate_user_input(req.message)
        except (ControlViolationError, ControlSteerError) as exc:
            reason = f"Agent Control: {exc}"
            logger.conclude(output=reason)
            concluded = True
            yield _sse({"type": "guardrail", "stage": "input", "reason": reason})
            yield _sse({"type": "done"})
            return
        except Exception:  # noqa: BLE001
            log.exception("Agent Control input check failed; allowing the turn.")

        collected: list[str] = []
        tool_calls: list[tuple[str, object, object]] = []
        pending_tools: dict = {}
        with rag_collection_scope(domain.collection):
            agent = build_agent(
                provider=req.provider, temperature=req.temperature, domain=domain.name
            )
            # Only the native logger traces this turn; silence the OTel instrumentor.
            suppress = otel_context.attach(
                otel_context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
            )
            try:
                async for event in agent.astream_events(
                    {"messages": [("user", req.message)]},
                    config={"configurable": {"thread_id": req.conversation_id}},
                    version="v2",
                ):
                    kind = event.get("event")
                    if kind == "on_chat_model_stream":
                        token = _chunk_text(event["data"]["chunk"])
                        if token:
                            collected.append(token)
                            yield _sse({"type": "token", "text": token})
                    elif kind == "on_tool_start":
                        name = event.get("name", "tool")
                        tin = event.get("data", {}).get("input")
                        pending_tools[event.get("run_id")] = (name, tin)
                        yield _sse({"type": "tool", "name": name, "input": tin})
                    elif kind == "on_tool_end":
                        name, tin = pending_tools.pop(event.get("run_id"), (None, None))
                        if name:
                            tool_calls.append(
                                (name, tin, event.get("data", {}).get("output"))
                            )
            finally:
                otel_context.detach(suppress)

        # Log tool calls (e.g. the bank customer DB query) as tool spans so the
        # DB step shows in the native trace alongside the control spans.
        for tname, tin, tout in tool_calls:
            try:
                logger.add_tool_span(
                    input=json.dumps(tin, default=str),
                    output=str(getattr(tout, "content", tout)),
                    name=tname,
                )
            except Exception:  # noqa: BLE001
                pass

        answer = "".join(collected)
        displayed_answer = answer
        if req.guardrails_enabled:
            gout = _guardrail.check_output(answer)
            displayed_answer = (
                gout.redacted_text if gout.redacted_text is not None else answer
            )
            if displayed_answer != answer:
                yield _sse({"type": "redacted", "text": displayed_answer})

        logger.add_llm_span(
            input=req.message,
            output=displayed_answer,
            model=cfg.model,
            name="Agent Chat",
        )
        logger.conclude(output=displayed_answer)
        concluded = True
    except Exception as exc:  # noqa: BLE001
        log.exception("agent error")
        if not concluded:
            try:
                logger.conclude(output=f"error: {exc}")
            except Exception:  # noqa: BLE001
                pass
        yield _sse({"type": "error", "message": str(exc)})
    finally:
        try:
            logger.flush()
        except Exception:  # noqa: BLE001
            pass
    yield _sse({"type": "done"})


async def _event_stream(req: ChatRequest):
    settings = get_settings()
    cfg = settings.providers().get(req.provider or settings.llm_provider)
    if cfg is None:
        yield _sse({"type": "error", "message": f"Unknown provider: {req.provider}"})
        yield _sse({"type": "done"})
        return

    domains = get_domains()
    domain_name = req.domain or DEFAULT_DOMAIN
    domain = domains.get(domain_name)
    if domain is None:
        yield _sse({"type": "error", "message": f"Unknown domain: {req.domain}"})
        yield _sse({"type": "done"})
        return

    # Per-domain tracing: native Galileo SDK for the configured domains (deepest
    # Agent Control integration -- rich control span + Controls tab); every other
    # domain stays on the OTLP -> Collector -> Galileo path (OTel adoption story).
    # When the OTel control sink is active, all domains stay on the OTel path.
    if (
        agent_control_active()
        and not otel_sink_active()
        and domain.name in settings.native_domains()
    ):
        async for chunk in _agent_control_stream(req, cfg, domain):
            yield chunk
        return

    # The scope remains active for the full generator lifetime, so all nested
    # OpenLLMetry spans inherit this turn root and the real provider identity.
    with active_genai_system_scope(cfg.genai_system), active_domain_scope(
        domain.name
    ), active_guardrails_scope(
        req.guardrails_enabled
    ), rag_collection_scope(
        domain.collection
    ), start_chat_span(
        conversation_id=req.conversation_id,
        provider_name=cfg.genai_system,
        model=cfg.model,
        input_text=req.message,
    ) as turn_span:
        turn_span.set_attribute("domain.name", domain.name)
        turn_span.set_attribute("app.guardrails.enabled", req.guardrails_enabled)
        yield _sse(
            {"type": "meta", "domain": domain.name, "collection": domain.collection}
        )

        # Agent Control (optional): evaluate the user input as a distinct child
        # step so both allowed and blocked turns show the control in the trace.
        if agent_control_active():
            with start_guardrail_span("input") as ac_span:
                ac_span.set_attribute("guardrail.provider", "galileo-agent-control")
                # This span evaluates the user message; recording it as the span
                # input also gives Galileo the turn input when the streamed LLM
                # child spans come back empty.
                set_turn_input(ac_span, req.message)
                # Nest Agent Control's OTLP control spans under this turn (no-op
                # unless the otel control sink is active).
                bind_otel_trace_context(turn_span)
                try:
                    await evaluate_user_input(req.message)
                    set_guardrail_result(ac_span, allowed=True)
                    ac_span.set_attribute(
                        "output.value", json.dumps({"action": "allow", "matched": False})
                    )
                except (ControlViolationError, ControlSteerError) as exc:
                    reason = f"Agent Control: {exc}"
                    set_guardrail_result(ac_span, allowed=False, reason=reason)
                    action = "steer" if isinstance(exc, ControlSteerError) else "deny"
                    ac_span.set_attribute(
                        "output.value",
                        json.dumps({"action": action, "matched": True, "reason": reason}),
                    )
                    set_turn_output(turn_span, reason)
                    turn_span.set_attribute("agent.result", "blocked_by_agent_control")
                    turn_span.set_status(Status(StatusCode.OK))
                    yield _sse({"type": "guardrail", "stage": "input", "reason": reason})
                    yield _sse({"type": "done"})
                    return
                except Exception:  # noqa: BLE001
                    set_guardrail_result(
                        ac_span, allowed=True, reason="control check errored; allowed"
                    )
                    log.exception("Agent Control input check failed; allowing the turn.")
                finally:
                    unbind_otel_trace_context()

        if req.guardrails_enabled:
            with start_guardrail_span("input") as guardrail_span:
                gin = _guardrail.check_input(req.message)
                set_guardrail_result(
                    guardrail_span, allowed=gin.allowed, reason=gin.reason
                )
            if not gin.allowed:
                turn_span.set_attribute("agent.result", "blocked")
                set_turn_output(turn_span, gin.reason)
                yield _sse({"type": "guardrail", "stage": "input", "reason": gin.reason})
                yield _sse({"type": "done"})
                return

        collected: list[str] = []
        agent = build_agent(
            provider=req.provider, temperature=req.temperature, domain=domain.name
        )

        # Retrieval runs inside the agent (pre_model_hook), so retriever, LLM,
        # and tool spans remain children of this complete chat-turn trace.
        try:
            async for event in agent.astream_events(
                {"messages": [("user", req.message)]},
                config={
                    "configurable": {"thread_id": req.conversation_id},
                    # Distinct from the "invoke_agent Agent Chat" turn-root span:
                    # the LangGraph instrumentor emits its own invoke_agent span
                    # named after run_name, and an identical name makes Galileo
                    # merge the two, taking input from the instrumentor's empty one.
                    "run_name": "Agent Graph",
                },
                version="v2",
            ):
                kind = event.get("event")
                if kind == "on_chat_model_stream":
                    token = _chunk_text(event["data"]["chunk"])
                    if token:
                        collected.append(token)
                        yield _sse({"type": "token", "text": token})
                elif kind == "on_tool_start":
                    yield _sse(
                        {
                            "type": "tool",
                            "name": event.get("name", "tool"),
                            "input": event.get("data", {}).get("input"),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            log.exception("agent error")
            turn_span.record_exception(exc)
            turn_span.set_attribute("error.type", type(exc).__name__)
            turn_span.set_status(Status(StatusCode.ERROR, str(exc)))
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse({"type": "done"})
            return

        answer = "".join(collected)
        displayed_answer = answer
        redacted = False
        if req.guardrails_enabled:
            with start_guardrail_span("output") as guardrail_span:
                gout = _guardrail.check_output(answer)
                displayed_answer = (
                    gout.redacted_text if gout.redacted_text is not None else answer
                )
                redacted = displayed_answer != answer
                set_guardrail_result(
                    guardrail_span,
                    allowed=gout.allowed,
                    reason=gout.reason,
                    redacted=redacted,
                )
        turn_span.set_attribute("agent.result", "completed")
        # Re-assert input: the streaming instrumentor overwrites gen_ai.input.messages
        # with an empty capture during the run, so restore it after the loop.
        set_turn_input(turn_span, req.message)
        set_turn_output(turn_span, displayed_answer)
        if redacted:
            yield _sse({"type": "redacted", "text": displayed_answer})

        yield _sse({"type": "done"})


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(_event_stream(req), media_type="text/event-stream")


# Serve the chat UI. The index page is returned with no-cache so layout changes
# always show on refresh; other static assets (if any) keep default caching.
_INDEX_HTML = _STATIC_DIR / "index.html"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_INDEX_HTML, headers={"Cache-Control": "no-cache"})


if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")
