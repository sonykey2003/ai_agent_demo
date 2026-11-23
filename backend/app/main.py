"""FastAPI entrypoint: streaming chat over an agent, plus provider listing.

The chat endpoint streams Server-Sent Events so the UI can render tokens, tool
calls, and guardrail notices as they happen. The static chat UI is served from
the same app to keep the demo to a single container.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry.trace import Span, Status, StatusCode
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile

from .agent.graph import assistant_node_name, build_agent
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
from .attachments import ocr
from .attachments.context import build_context
from .attachments.processing import ACCEPTED_TYPES, process_upload
from .attachments.store import Attachment, AttachmentError, get_store
from .config import get_settings
from .domains import DEFAULT_DOMAIN, get_domains
from .providers import make_chat_model, provider_supports_vision
from .rag.ingest import ingest_all
from .telemetry.otel import (
    active_domain_scope,
    active_genai_system_scope,
    apm_only_scope,
    set_guardrail_result,
    set_turn_attachments,
    set_turn_input,
    set_turn_output,
    setup_telemetry,
    shutdown_telemetry,
    start_chat_span,
    start_guardrail_span,
    start_server_span,
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


def _warm_ocr() -> None:
    try:
        ocr.get_engine()
    except Exception:  # noqa: BLE001
        log.exception("OCR engine failed to load; image text recognition will fail.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_telemetry(_app)
    setup_agent_control()
    if get_settings().attachment_ocr_enabled:
        # Load the OCR models in the background so the first image upload is fast.
        asyncio.get_running_loop().run_in_executor(None, _warm_ocr)
    try:
        await _initialize_rag()
        yield
    finally:
        shutdown_telemetry()


# FastAPI >=0.142 otherwise claims the global OTel providers before lifespan runs setup_telemetry().
app = FastAPI(
    title="Vendor-Agnostic AI Agent Chat Demo",
    lifespan=lifespan,
    telemetry={
        "auto_configure": False,
        "tracing": False,
        "metrics": False,
        "logs": False,
    },
)

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
    # IDs returned by POST /attachments for this conversation.
    attachments: list[str] = Field(default_factory=list, max_length=20)


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


_ROLES = {"human": "user", "ai": "assistant"}


def _content_text(content) -> str:
    """Message content as text; image blocks become a placeholder, never base64."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for piece in content:
            if isinstance(piece, dict):
                kind = piece.get("type")
                if kind == "text":
                    parts.append(piece.get("text") or "")
                elif kind in ("image_url", "image"):
                    parts.append("[image]")
                else:
                    parts.append(f"[{kind}]")
            else:
                parts.append(str(piece))
        return "\n".join(p for p in parts if p)
    return str(content)


def _llm_message(m) -> dict | None:
    """One LangChain message in Galileo's shape.

    Returns None when the message would carry neither content nor tool calls:
    Galileo rejects that with a 422 that silently drops the *whole* trace, and
    it is exactly the shape of a tool-calling assistant turn.
    """
    kind = getattr(m, "type", "") or "user"
    content = getattr(m, "content", m)
    msg: dict = {
        "role": _ROLES.get(kind, kind),
        "content": _content_text(content),
    }
    calls = [
        {
            "id": c.get("id") or "",
            "function": {
                "name": c.get("name") or "",
                "arguments": json.dumps(c.get("args") or {}, default=str),
            },
        }
        for c in (getattr(m, "tool_calls", None) or [])
    ]
    if calls:
        msg["tool_calls"] = calls
    tool_call_id = getattr(m, "tool_call_id", None)
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    if not msg["content"] and not calls:
        return None
    return msg


def _llm_messages(raw) -> list[dict] | None:
    """Flatten the messages the model actually saw, so context-grounding metrics
    score against the injected passages rather than the bare user question."""
    if not raw:
        return None
    msgs = raw[0] if isinstance(raw[0], list) else raw
    out = [d for d in (_llm_message(m) for m in msgs) if d]
    return out or None


def _span_times(started_ns: int | None) -> dict:
    """Real start/duration for a step; without them Galileo stamps the span at
    flush time and the trace renders out of order against the control spans."""
    if not started_ns:
        return {}
    return {
        "created_at": datetime.fromtimestamp(started_ns / 1e9, tz=timezone.utc),
        "duration_ns": time.time_ns() - started_ns,
    }


def _log_retriever_span(logger, output: dict, domain, fallback: str, started) -> None:
    """Retrieval is a graph node, not a tool, so the native path has to re-declare
    it from the state the node returns or RAG scorers get no documents."""
    try:
        logger.add_retriever_span(
            input=output.get("retrieval_query") or fallback,
            output=output.get("documents") or [],
            name=f"retrieve {domain.collection}",
            metadata={"collection": domain.collection},
            **_span_times(started),
        )
    except Exception:  # noqa: BLE001
        log.exception("could not log retriever span")


def _log_llm_span(logger, model_input, output, model: str, started, name: str) -> None:
    try:
        logger.add_llm_span(
            input=model_input,
            output=_llm_message(output) or "",
            model=model,
            # Named for the graph node, not "Agent Chat" — that is the turn-root
            # agent name, and reusing it made every reasoning step look identical.
            name=name,
            **_span_times(started),
        )
    except Exception:  # noqa: BLE001
        log.exception("could not log llm span")


def _log_tool_span(logger, name: str, tool_input, output, started) -> None:
    try:
        logger.add_tool_span(
            input=json.dumps(tool_input, default=str),
            output=str(getattr(output, "content", output)),
            name=name,
            **_span_times(started),
        )
    except Exception:  # noqa: BLE001
        log.exception("could not log tool span")


_STEER_REVISION_PROMPT = (
    "You are revising an assistant reply so it complies with a policy control.\n"
    "Policy guidance: {guidance}\n"
    "Rewrite the reply to satisfy that guidance while keeping every compliant "
    "detail and the original tone. Mask or drop only the offending content. "
    "Reply with the revised text alone -- no preamble, no explanation."
)


async def _revise_for_steer(
    answer: str, guidance: str, req: ChatRequest
) -> str | None:
    """Rewrite `answer` per a steer control's steering context, then re-check it.

    Returns None when the control gave no guidance or the rewrite still trips a
    control, so the caller falls back to withholding the answer.
    """
    if not guidance:
        return None
    model = make_chat_model(provider=req.provider, temperature=req.temperature)
    result = await model.ainvoke(
        [
            ("system", _STEER_REVISION_PROMPT.format(guidance=guidance)),
            ("user", answer),
        ]
    )
    revised = _chunk_text(result).strip()
    if not revised:
        return None
    try:
        await evaluate_assistant_output(revised)
    except (ControlSteerError, ControlViolationError):
        return None
    return revised


async def _resolve_output_control(
    answer: str, req: ChatRequest
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
        revised = await _revise_for_steer(answer, info["detail"], req)
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
            {
                "id": p.name,
                "label": p.label,
                "model": p.model,
                "supports_vision": provider_supports_vision(p),
            }
            for p in settings.providers().values()
        ],
    }


# Room for multipart boundaries and form fields on top of the file itself.
_MULTIPART_OVERHEAD = 1024 * 1024
_UPLOAD_CHUNK = 1024 * 1024


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    """Read an upload, stopping as soon as it exceeds the limit."""
    chunks, total = [], 0
    while chunk := await upload.read(_UPLOAD_CHUNK):
        total += len(chunk)
        if total > limit:
            raise AttachmentError(
                413, f"File is larger than {limit // (1024 * 1024)} MB."
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.get("/attachments/limits")
def attachment_limits() -> dict:
    settings = get_settings()
    return {
        "max_mb": settings.attachment_max_mb,
        "max_files": settings.attachment_max_files,
        "accepted": ACCEPTED_TYPES,
        "ocr_enabled": settings.attachment_ocr_enabled,
        "ocr_languages": ocr.LANGUAGES,
        "ocr_max_pages": settings.attachment_ocr_max_pages,
        "max_pdf_pages": settings.attachment_max_pdf_pages,
    }


@app.post("/attachments")
async def upload_attachment(request: Request) -> JSONResponse:
    """Upload one file (multipart fields: file, conversation_id)."""
    settings = get_settings()
    limit = settings.attachment_max_mb * 1024 * 1024
    # Upload traces are plain APM traces; keep them out of the LLM backends.
    with apm_only_scope(), start_server_span(method="POST", route="/attachments") as span:
        form = None
        try:
            declared = int(request.headers.get("content-length") or 0)
            if declared > limit + _MULTIPART_OVERHEAD:
                raise AttachmentError(
                    413, f"File is larger than {settings.attachment_max_mb} MB."
                )
            form = await request.form(max_files=1, max_fields=4)
            upload = form.get("file")
            conversation_id = str(form.get("conversation_id") or "").strip()
            if not isinstance(upload, UploadFile):
                raise AttachmentError(422, "Missing 'file' field.")
            if not conversation_id or len(conversation_id) > 128:
                raise AttachmentError(422, "Missing or invalid 'conversation_id'.")
            span.set_attribute("gen_ai.conversation.id", conversation_id)
            span.set_attribute("session.id", conversation_id)
            data = await _read_limited(upload, limit)
            span.set_attribute("attachment.size_bytes", len(data))
            attachment = await asyncio.to_thread(
                process_upload, conversation_id, upload.filename or "upload", data
            )
        except AttachmentError as exc:
            span.set_attribute("http.response.status_code", exc.status_code)
            span.set_attribute("attachment.error", exc.message)
            log.info("attachment rejected (%d): %s", exc.status_code, exc.message)
            return JSONResponse({"error": exc.message}, status_code=exc.status_code)
        except Exception as exc:  # noqa: BLE001
            log.exception("attachment processing failed")
            span.record_exception(exc)
            span.set_attribute("error.type", type(exc).__name__)
            span.set_attribute("http.response.status_code", 500)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            return JSONResponse(
                {"error": f"Could not process the file: {exc}"}, status_code=500
            )
        finally:
            if form is not None:
                await form.close()

        span.set_attribute("attachment.id", attachment.id)
        span.set_attribute("attachment.kind", attachment.kind)
        span.set_attribute("attachment.mime_type", attachment.mime_type)
        span.set_attribute("attachment.chars", len(attachment.text))
        if attachment.meta.get("pages"):
            span.set_attribute("attachment.pages", attachment.meta["pages"])
        return JSONResponse(attachment.descriptor())


@app.delete("/attachments/{attachment_id}")
def delete_attachment(attachment_id: str, conversation_id: str) -> JSONResponse:
    if not get_store().delete(conversation_id, attachment_id):
        return JSONResponse({"error": "Attachment not found."}, status_code=404)
    return JSONResponse({"deleted": attachment_id})


@app.get("/domains")
def domains() -> dict:
    return {
        "default": DEFAULT_DOMAIN,
        "domains": [
            {"id": d.name, "label": d.label} for d in get_domains().values()
        ],
    }


def _record_apm_outcome(span: Span | None, output: str, result: str) -> None:
    if span is None:
        return
    set_turn_output(span, output)
    span.set_attribute("agent.result", result)


async def _agent_control_stream(
    req: ChatRequest, cfg, domain, apm_span: Span | None = None
):
    """Agent Control turn traced by the native Galileo logger (reference pattern).

    The logger sets the trace input/output explicitly and enable_agent_control()
    bridges the @control span into the same trace, so input, the control decision,
    and output all render. The caller runs this inside apm_only_scope(), so the
    OTel spans of the turn reach Splunk APM but not Galileo, and the logger owns
    the one Galileo trace. ``apm_span`` is the OTel turn span.
    """
    logger = get_ac_logger(domain.name)
    yield _sse(
        {"type": "meta", "domain": domain.name, "collection": domain.collection}
    )

    otel_trace_id = apm_span.get_span_context().trace_id if apm_span else None
    trace = logger.start_trace(
        input=req.message,
        name="Agent Chat",
        metadata=(
            {"otel_trace_id": format(otel_trace_id, "032x")} if otel_trace_id else None
        ),
    )
    # start_trace always mints a random ID. In batch mode nothing is sent until
    # flush(), so adopting the (UUIDv4-compliant) OTel trace ID here makes the
    # Galileo and Splunk APM trace IDs identical. Distributed mode streams the
    # trace immediately, so it keeps its own ID and relies on the metadata link.
    if otel_trace_id and getattr(logger, "mode", None) == "batch":
        trace.id = UUID(int=otel_trace_id)
    galileo_trace_id = getattr(trace, "id", None)
    if apm_span is not None and galileo_trace_id:
        apm_span.set_attribute("galileo.trace_id", str(galileo_trace_id))
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
            _record_apm_outcome(apm_span, reason, "blocked_by_agent_control")
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
        pending_tools: dict = {}
        pending_llm: dict = {}
        pending_retrieval: dict = {}
        with rag_collection_scope(domain.collection):
            agent = build_agent(
                provider=req.provider, temperature=req.temperature, domain=domain.name
            )
            # Spans are logged as each step ENDS, interleaved with the control
            # spans the Agent Control bridge writes live, so the trace reads in
            # execution order: retrieve -> llm -> control -> tool -> llm.
            async for event in agent.astream_events(
                {"messages": [("user", req.message)]},
                config={"configurable": {"thread_id": req.conversation_id}},
                version="v2",
            ):
                kind = event.get("event")
                run_id = event.get("run_id")
                name = event.get("name", "")
                if kind == "on_chain_start" and name == "retrieve_context":
                    pending_retrieval[run_id] = time.time_ns()
                elif kind == "on_chain_end" and name == "retrieve_context":
                    started = pending_retrieval.pop(run_id, None)
                    out = event.get("data", {}).get("output") or {}
                    _log_retriever_span(logger, out, domain, req.message, started)
                elif kind == "on_chat_model_start":
                    pending_llm[run_id] = (
                        _llm_messages(
                            event.get("data", {}).get("input", {}).get("messages")
                        ),
                        time.time_ns(),
                    )
                elif kind == "on_chat_model_stream":
                    token = _chunk_text(event["data"]["chunk"])
                    if token:
                        collected.append(token)
                        yield _sse({"type": "token", "text": token})
                elif kind == "on_chat_model_end":
                    msgs, started = pending_llm.pop(run_id, (None, None))
                    _log_llm_span(
                        logger,
                        msgs or req.message,
                        event.get("data", {}).get("output"),
                        cfg.model,
                        started,
                        assistant_node_name(domain.name),
                    )
                elif kind == "on_tool_start":
                    tin = event.get("data", {}).get("input")
                    pending_tools[run_id] = (name or "tool", tin, time.time_ns())
                    yield _sse({"type": "tool", "name": name, "input": tin})
                elif kind == "on_tool_end":
                    tname, tin, started = pending_tools.pop(
                        run_id, (None, None, None)
                    )
                    if tname:
                        _log_tool_span(
                            logger,
                            tname,
                            tin,
                            event.get("data", {}).get("output"),
                            started,
                        )

        answer = "".join(collected)
        displayed_answer = answer

        # Output-side Agent Control. Galileo owns the outcome end to end (no built-in
        # regex redaction): a steer rewrites the answer with the control's steering
        # context, a deny withholds it.
        if agent_control_active():
            control_info: dict | None = None
            try:
                displayed_answer, control_info = await _resolve_output_control(
                    answer, req
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

        logger.conclude(output=displayed_answer)
        concluded = True
        _record_apm_outcome(apm_span, displayed_answer, "completed")
    except Exception as exc:  # noqa: BLE001
        log.exception("agent error")
        if apm_span is not None:
            apm_span.record_exception(exc)
            apm_span.set_attribute("error.type", type(exc).__name__)
            apm_span.set_status(Status(StatusCode.ERROR, str(exc)))
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


_DEFAULT_ATTACHMENT_PROMPT = "Please review the attached file(s)."


def _prepare_attachments(req: ChatRequest, cfg) -> tuple[list[Attachment], dict | None]:
    """Validate and bind this turn's attachments.

    Returns the attachments and the SSE notice describing how each reaches the
    model. Raises AttachmentError before anything is bound when the turn can't
    proceed (unknown id, too many files, image a text-only model can't use).
    """
    settings = get_settings()
    store = get_store()
    if not req.attachments:
        store.begin_turn(req.conversation_id, [])
        return [], None
    if len(req.attachments) > settings.attachment_max_files:
        raise AttachmentError(
            422, f"Attach at most {settings.attachment_max_files} files per message."
        )
    vision = False
    for attachment_id in req.attachments:
        item = store.get(req.conversation_id, attachment_id)
        if item is None:
            raise AttachmentError(404, "An attachment expired or was removed; attach it again.")
        if item.kind == "image":
            vision = provider_supports_vision(cfg)
            if not vision and not item.text.strip():
                raise AttachmentError(
                    422,
                    f"{cfg.model} can't see images and no text was recognized in "
                    f"{item.name}. Choose a vision model (e.g. OpenAI or a vision "
                    "Ollama model).",
                )
    items = store.begin_turn(req.conversation_id, list(req.attachments))
    budget = (
        settings.attachment_inline_max_chars_local
        if cfg.genai_system == "ollama"
        else settings.attachment_inline_max_chars
    )
    _, truncated = build_context(
        store.for_conversation(req.conversation_id), budget, vision=vision
    )
    notice = {
        "type": "attachments",
        "model": cfg.model,
        "vision": vision,
        "items": [
            {
                "name": a.name,
                "kind": a.kind,
                "delivery": (
                    ("image+ocr" if vision else "ocr_only") if a.kind == "image" else "text"
                ),
                "pages": a.meta.get("pages"),
                "chars": len(a.text),
                "ocr": a.meta.get("ocr"),
                "truncated": truncated.get(a.id, False),
            }
            for a in items
        ],
    }
    return items, notice


def _attachment_delivery(items: list[Attachment], vision: bool) -> str:
    if any(a.kind == "image" for a in items):
        return "image+ocr" if vision else "ocr_only"
    return "text"


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

    try:
        attachments, attachment_notice = await asyncio.to_thread(
            _prepare_attachments, req, cfg
        )
    except AttachmentError as exc:
        yield _sse({"type": "error", "message": exc.message})
        yield _sse({"type": "done"})
        return
    if attachments and not req.message.strip():
        req = req.model_copy(update={"message": _DEFAULT_ATTACHMENT_PROMPT})
    attachment_refs = [a.descriptor() for a in attachments]

    def _mark_attachments(span: Span) -> None:
        if attachments:
            set_turn_attachments(
                span,
                attachment_refs,
                _attachment_delivery(attachments, attachment_notice["vision"]),
            )

    # Per-domain tracing: native Galileo SDK for the configured domains (deepest
    # Agent Control integration -- rich control span + Controls tab), each into its
    # own log stream; every other domain stays on the OTLP -> Collector -> Galileo
    # path (OTel adoption story). When the OTel control sink is active, all domains
    # stay on the OTel path.
    if (
        agent_control_active()
        and not otel_sink_active()
        and domain.name in settings.native_domains()
        and get_ac_logger(domain.name) is not None
    ):
        # The native logger owns this turn's Galileo trace. Every OTel span of the
        # turn (entry, retrieval/DB, LLM, tools) is marked APM-only, so Splunk APM
        # gets the full tree while the Collector keeps it out of Galileo/Langfuse.
        with active_genai_system_scope(cfg.genai_system), active_domain_scope(
            domain.name
        ), apm_only_scope(), start_server_span(
            method="POST", route="/chat"
        ), start_chat_span(
            conversation_id=req.conversation_id,
            provider_name=cfg.genai_system,
            model=cfg.model,
            input_text=req.message,
            attachments=attachment_refs,
        ) as turn_span:
            turn_span.set_attribute("domain.name", domain.name)
            _mark_attachments(turn_span)
            if attachment_notice:
                yield _sse(attachment_notice)
            async for chunk in _agent_control_stream(
                req, cfg, domain, apm_span=turn_span
            ):
                yield chunk
        return

    # The scope remains active for the full generator lifetime, so all nested
    # OpenLLMetry spans inherit this turn root and the real provider identity.
    with active_genai_system_scope(cfg.genai_system), active_domain_scope(
        domain.name
    ), rag_collection_scope(
        domain.collection
    ), start_server_span(
        method="POST", route="/chat"
    ), start_chat_span(
        conversation_id=req.conversation_id,
        provider_name=cfg.genai_system,
        model=cfg.model,
        input_text=req.message,
        attachments=attachment_refs,
    ) as turn_span:
        turn_span.set_attribute("domain.name", domain.name)
        _mark_attachments(turn_span)
        yield _sse(
            {"type": "meta", "domain": domain.name, "collection": domain.collection}
        )
        if attachment_notice:
            yield _sse(attachment_notice)

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
        set_turn_input(turn_span, req.message, attachment_refs)
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
