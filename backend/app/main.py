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
from datetime import datetime, timezone
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
from .agent.agent_control import (
    ControlSteerError,
    ControlViolationError,
    bind_otel_trace_context,
    control_block_message,
    describe_control_error,
    evaluate_assistant_output,
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
from .providers import make_chat_model
from .rag.ingest import ingest_all
from .telemetry.otel import (
    active_domain_scope,
    active_genai_system_scope,
    mute_otel_scope,
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

_STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str
    provider: str | None = None
    domain: str | None = None
    temperature: float | None = None
    conversation_id: str = Field(
        default_factory=lambda: str(uuid4()), min_length=1, max_length=128
    )


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _elapsed_ns(start: datetime | None, end: datetime | None) -> int | None:
    if start is None or end is None:
        return None
    return max(int((end - start).total_seconds() * 1_000_000_000), 0)


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


_STEER_REVISION_PROMPT = (
    "You are revising an assistant reply so it complies with a policy control.\n"
    "Policy guidance: {guidance}\n"
    "Rewrite the reply to satisfy that guidance while keeping every compliant "
    "detail and the original tone. Mask or drop only the offending content. "
    "Reply with the revised text alone -- no preamble, no explanation."
)


async def _revise_for_steer(
    answer: str,
    guidance: str,
    req: ChatRequest,
    *,
    suppress_otel: bool,
    native_logger=None,
    model_name: str | None = None,
) -> str | None:
    """Rewrite `answer` per a steer control's steering context, then re-check it.

    Returns None when the control gave no guidance or the rewrite still trips a
    control, so the caller falls back to withholding the answer.
    """
    if not guidance:
        return None
    model = make_chat_model(provider=req.provider, temperature=req.temperature)
    token = (
        otel_context.attach(
            otel_context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
        )
        if suppress_otel
        else None
    )
    started = _utcnow()
    try:
        result = await model.ainvoke(
            [
                ("system", _STEER_REVISION_PROMPT.format(guidance=guidance)),
                ("user", answer),
            ]
        )
    finally:
        if token is not None:
            otel_context.detach(token)
    revised = _chunk_text(result).strip()
    # Without this the re-check below reads as a second control set with no cause.
    if native_logger is not None:
        try:
            native_logger.add_llm_span(
                input=answer,
                output=revised,
                model=model_name,
                name="Steer revision",
                created_at=started,
                duration_ns=_elapsed_ns(started, _utcnow()),
                metadata={"steering_context": guidance},
            )
        except Exception:  # noqa: BLE001
            log.exception("Could not log the steer-revision span.")
    if not revised:
        return None
    try:
        await evaluate_assistant_output(revised)
    except (ControlSteerError, ControlViolationError):
        return None
    return revised


async def _resolve_output_control(
    answer: str,
    req: ChatRequest,
    *,
    suppress_otel: bool = False,
    native_logger=None,
    model_name: str | None = None,
) -> tuple[str, dict | None]:
    """Apply the output-side control: steer revises the answer, deny withholds it.

    Returns (text to display, control info) with info None when nothing fired.
    Non-control failures propagate so the caller can fail open.
    """
    try:
        await evaluate_assistant_output(answer)
        return answer, None
    except ControlSteerError as exc:
        info = describe_control_error(exc)
        revised = await _revise_for_steer(
            answer,
            info["detail"],
            req,
            suppress_otel=suppress_otel,
            native_logger=native_logger,
            model_name=model_name,
        )
        info["revised"] = revised is not None
        if revised is not None:
            return revised, info
        return control_block_message(info), info
    except ControlViolationError as exc:
        info = describe_control_error(exc)
        info["revised"] = False
        return control_block_message(info), info


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
            info = describe_control_error(exc)
            reason = info["reason"]
            logger.conclude(output=reason)
            concluded = True
            yield _sse(
                {
                    "type": "guardrail",
                    "stage": "input",
                    "reason": reason,
                    "control": info["control"],
                    "action": info["action"],
                }
            )
            yield _sse({"type": "done"})
            return
        except Exception:  # noqa: BLE001
            log.exception("Agent Control input check failed; allowing the turn.")

        collected: list[str] = []
        tool_calls: list[tuple[str, object, object, object, object]] = []
        pending_tools: dict = {}
        retrieval: dict | None = None
        retrieval_started = retrieval_ended = None
        llm_started = llm_ended = None
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
                    if kind == "on_chain_start" and event.get("name") == "retrieve_context":
                        retrieval_started = _utcnow()
                    elif kind == "on_chain_end" and event.get("name") == "retrieve_context":
                        # OTel is suppressed here, so the graph's retriever span never
                        # ships; carry the evidence out and log it natively below.
                        retrieval = event.get("data", {}).get("output") or None
                        retrieval_ended = _utcnow()
                    elif kind == "on_chat_model_start":
                        # Reassigned per model call, so the last one wins: that is the
                        # generation that produced the answer, after any tool round trip.
                        llm_started = _utcnow()
                    elif kind == "on_chat_model_end":
                        llm_ended = _utcnow()
                    elif kind == "on_chat_model_stream":
                        token = _chunk_text(event["data"]["chunk"])
                        if token:
                            # Buffered, never streamed: an output-side control can only
                            # score a finished answer, so nothing leaves the server until
                            # it has cleared.
                            if not collected:
                                yield _sse({"type": "buffering"})
                            collected.append(token)
                    elif kind == "on_tool_start":
                        name = event.get("name", "tool")
                        tin = event.get("data", {}).get("input")
                        pending_tools[event.get("run_id")] = (name, tin, _utcnow())
                        yield _sse({"type": "tool", "name": name, "input": tin})
                    elif kind == "on_tool_end":
                        name, tin, tstart = pending_tools.pop(
                            event.get("run_id"), (None, None, None)
                        )
                        if name:
                            tool_calls.append(
                                (
                                    name,
                                    tin,
                                    event.get("data", {}).get("output"),
                                    tstart,
                                    _utcnow(),
                                )
                            )
            finally:
                otel_context.detach(suppress)

        # Retrieval ran inside the graph under OTel suppression, so replay it as a
        # native retriever span -- this is what Galileo's RAG scorers read.
        if retrieval and retrieval.get("documents"):
            try:
                logger.add_retriever_span(
                    input=retrieval.get("retrieval_query") or req.message,
                    output=retrieval["documents"],
                    name="retrieve knowledge_base",
                    created_at=retrieval_started,
                    duration_ns=_elapsed_ns(retrieval_started, retrieval_ended),
                    metadata={"collection": domain.collection},
                )
            except Exception:  # noqa: BLE001
                log.exception("Could not log the retriever span.")

        # Log tool calls (e.g. the bank customer DB query) as tool spans so the
        # DB step shows in the native trace alongside the control spans.
        for tname, tin, tout, tstart, tend in tool_calls:
            try:
                logger.add_tool_span(
                    input=json.dumps(tin, default=str),
                    output=str(getattr(tout, "content", tout)),
                    name=tname,
                    created_at=tstart,
                    duration_ns=_elapsed_ns(tstart, tend),
                )
            except Exception:  # noqa: BLE001
                pass

        answer = "".join(collected)
        displayed_answer = answer
        control_info: dict | None = None

        # Logged before the output control so the trace reads retrieve -> tool ->
        # llm -> output control, matching what actually happened.
        logger.add_llm_span(
            input=req.message,
            output=answer,
            model=cfg.model,
            name="Agent Chat",
            created_at=llm_started,
            duration_ns=_elapsed_ns(llm_started, llm_ended or _utcnow()),
        )

        # Output-side Agent Control. Galileo owns the outcome end to end (no built-in
        # regex redaction): a steer rewrites the answer with the control's steering
        # context, a deny withholds it.
        if agent_control_active():
            try:
                displayed_answer, control_info = await _resolve_output_control(
                    answer,
                    req,
                    suppress_otel=True,
                    native_logger=logger,
                    model_name=cfg.model,
                )
            except Exception:  # noqa: BLE001
                log.exception("Agent Control output check failed; showing answer as-is.")
            if control_info:
                yield _sse(
                    {
                        "type": "redacted",
                        "text": displayed_answer,
                        "source": "galileo-agent-control",
                        "control": control_info["control"],
                        "action": control_info["action"],
                        "revised": control_info["revised"],
                    }
                )
                yield _sse(
                    {
                        "type": "guardrail",
                        "stage": "output",
                        "reason": control_info["reason"],
                        "source": "galileo-agent-control",
                        "control": control_info["control"],
                        "action": control_info["action"],
                        "revised": control_info["revised"],
                    }
                )

        if control_info is None:
            yield _sse({"type": "answer", "text": displayed_answer})

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
        # The native logger owns this turn's trace; muting OTel stops a second,
        # LLM-less "invoke_agent LangGraph" trace landing beside it.
        with mute_otel_scope(), active_domain_scope(domain.name):
            async for chunk in _agent_control_stream(req, cfg, domain):
                yield chunk
        return

    # The scope remains active for the full generator lifetime, so all nested
    # OpenLLMetry spans inherit this turn root and the real provider identity.
    with active_genai_system_scope(cfg.genai_system), active_domain_scope(
        domain.name
    ), rag_collection_scope(
        domain.collection
    ), start_chat_span(
        conversation_id=req.conversation_id,
        provider_name=cfg.genai_system,
        model=cfg.model,
        input_text=req.message,
    ) as turn_span:
        turn_span.set_attribute("domain.name", domain.name)
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
                        "output.value",
                        json.dumps({"action": "allow", "matched": False}),
                    )
                except (ControlViolationError, ControlSteerError) as exc:
                    info = describe_control_error(exc)
                    reason = info["reason"]
                    set_guardrail_result(ac_span, allowed=False, reason=reason)
                    ac_span.set_attribute(
                        "output.value",
                        json.dumps(
                            {
                                "action": info["action"],
                                "matched": True,
                                "reason": reason,
                            }
                        ),
                    )
                    set_turn_output(turn_span, reason)
                    turn_span.set_attribute("agent.result", "blocked_by_agent_control")
                    turn_span.set_status(Status(StatusCode.OK))
                    yield _sse(
                        {
                            "type": "guardrail",
                            "stage": "input",
                            "reason": reason,
                            "control": info["control"],
                            "action": info["action"],
                        }
                    )
                    yield _sse({"type": "done"})
                    return
                except Exception:  # noqa: BLE001
                    set_guardrail_result(
                        ac_span, allowed=True, reason="control check errored; allowed"
                    )
                    log.exception("Agent Control input check failed; allowing the turn.")
                finally:
                    unbind_otel_trace_context()

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
        redacted_control: dict | None = None
        # Output-side Agent Control: Galileo owns the outcome. A steer rewrites the
        # answer with the control's steering context, a deny withholds it.
        if agent_control_active():
            with start_guardrail_span("output") as ac_out:
                ac_out.set_attribute("guardrail.provider", "galileo-agent-control")
                bind_otel_trace_context(turn_span)
                control_info: dict | None = None
                try:
                    displayed_answer, control_info = await _resolve_output_control(
                        answer, req
                    )
                    if control_info is None:
                        set_guardrail_result(ac_out, allowed=True)
                    else:
                        set_guardrail_result(
                            ac_out,
                            allowed=False,
                            reason=control_info["reason"],
                            redacted=True,
                        )
                except Exception:  # noqa: BLE001
                    set_guardrail_result(
                        ac_out, allowed=True, reason="control check errored; allowed"
                    )
                    log.exception("Agent Control output check failed; allowing the turn.")
                finally:
                    unbind_otel_trace_context()
                if control_info:
                    redacted = True
                    redacted_control = {
                        "control": control_info["control"],
                        "action": control_info["action"],
                        "revised": control_info["revised"],
                    }
                    yield _sse(
                        {
                            "type": "guardrail",
                            "stage": "output",
                            "reason": control_info["reason"],
                            "source": "galileo-agent-control",
                            **redacted_control,
                        }
                    )
        turn_span.set_attribute("agent.result", "completed")
        # Re-assert input: the streaming instrumentor overwrites gen_ai.input.messages
        # with an empty capture during the run, so restore it after the loop.
        set_turn_input(turn_span, req.message)
        set_turn_output(turn_span, displayed_answer)
        if redacted:
            payload = {
                "type": "redacted",
                "text": displayed_answer,
                "source": "galileo-agent-control",
            }
            if redacted_control:
                payload.update(redacted_control)
            yield _sse(payload)

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
