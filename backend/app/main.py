"""FastAPI entrypoint: streaming chat over an agent, plus provider listing.

The chat endpoint streams Server-Sent Events so the UI can render tokens, tool
calls, and guardrail notices as they happen. The static chat UI is served from
the same app to keep the demo to a single container.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent.graph import build_agent
from .agent.guardrails import get_guardrail
from .config import get_settings
from .telemetry.otel import set_active_genai_system, setup_telemetry

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Vendor-Agnostic AI Agent Chat Demo")
setup_telemetry(app)

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
    temperature: float | None = None


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


async def _event_stream(req: ChatRequest):
    # Input guardrail.
    gin = _guardrail.check_input(req.message)
    if not gin.allowed:
        yield _sse({"type": "guardrail", "stage": "input", "reason": gin.reason})
        yield _sse({"type": "done"})
        return

    # Label this request's LLM spans with the real provider (NVIDIA NIM / Ollama /
    # OpenAI) instead of the shared OpenAI-compatible client class.
    settings = get_settings()
    cfg = settings.providers().get(req.provider or settings.llm_provider)
    if cfg is not None:
        set_active_genai_system(cfg.genai_system)

    collected: list[str] = []
    agent = build_agent(provider=req.provider, temperature=req.temperature)

    # Retrieval runs inside the agent (pre_model_hook), so the retriever span,
    # LLM spans, and tool spans all sit in one trace rooted at invoke_agent.
    try:
        async for event in agent.astream_events({"messages": [("user", req.message)]}, version="v2"):
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
        yield _sse({"type": "error", "message": str(exc)})
        yield _sse({"type": "done"})
        return

    # Output guardrail (PII redaction).
    answer = "".join(collected)
    gout = _guardrail.check_output(answer)
    if gout.redacted_text is not None and gout.redacted_text != answer:
        yield _sse({"type": "redacted", "text": gout.redacted_text})

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
