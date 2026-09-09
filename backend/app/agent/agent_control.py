"""Optional Galileo Agent Control integration (opt-in; native SDK, not OTel).

Agent Control centrally evaluates LLM inputs/outputs (prompt injection, PII,
toxicity, ...) and can allow / steer / deny a step. This module keeps it fully
optional: when ``AGENT_CONTROL_ENABLED`` is false or the SDK is not installed,
``control`` is a no-op decorator and ``setup_agent_control`` does nothing, so the
vendor-neutral default is unchanged.

Enable it by setting in the environment / .env:
    AGENT_CONTROL_ENABLED=true
    AGENT_CONTROL_URL=https://agent-control.<env>.galileocloud.io
    GALILEO_API_KEY / GALILEO_PROJECT / GALILEO_LOG_STREAM   (already present)
and creating a control in the Galileo console bound to that log stream.
"""

from __future__ import annotations

import logging
import os
import uuid
from urllib.parse import urlparse

from ..config import get_settings

log = logging.getLogger(__name__)

try:  # The Agent Control SDK is an optional dependency.
    import agent_control as _ac
    from agent_control import ControlSteerError, ControlViolationError
except Exception:  # noqa: BLE001
    _ac = None

    class ControlViolationError(Exception):  # type: ignore[no-redef]
        """Fallback when the Agent Control SDK is not installed (deny action)."""

        control_name: str = "unknown"
        message: str = ""

    class ControlSteerError(Exception):  # type: ignore[no-redef]
        """Fallback when the Agent Control SDK is not installed (steer action)."""

        control_name: str = "unknown"
        message: str = ""
        steering_context: str = ""


# Decided once at import: controls only apply when explicitly enabled AND the SDK
# is available. The decorated step functions below bind against this.
_ENABLED = bool(get_settings().agent_control_enabled) and _ac is not None
_initialized = False
_logger = None  # native GalileoLogger that owns Agent Control turn traces
_otel_sink = False  # True when control spans are emitted over OTLP (otel sink)


def control(*args, **kwargs):
    """Real ``agent_control.control`` when enabled+available; else no-op passthrough."""
    if _ENABLED:
        return _ac.control(*args, **kwargs)

    def _decorator(fn):
        return fn

    return _decorator


def is_active() -> bool:
    """True only after a successful init, so callers can skip the network hop."""
    return _ENABLED and _initialized


def get_ac_logger():
    """Native Galileo logger that owns Agent Control turn traces (or None)."""
    return _logger if is_active() else None


def otel_sink_active() -> bool:
    """True when control spans are emitted over OTLP nested in the app trace."""
    return is_active() and _otel_sink


def bind_otel_trace_context(span) -> None:
    """Nest Agent Control's OTLP control spans under the given OTel span."""
    if not otel_sink_active():
        return
    ctx = span.get_span_context()
    # Galileo normalizes collector-ingested OTel trace_ids to a UUIDv4 (forcing
    # the version/variant bits). The control span goes direct-OTLP, so hand it the
    # same normalized id or the two land in separate traces.
    tid = uuid.UUID(int=ctx.trace_id, version=4).hex
    sid = format(ctx.span_id, "016x")
    _ac.set_trace_context_provider(lambda: {"trace_id": tid, "span_id": sid})


def unbind_otel_trace_context() -> None:
    """Clear the Agent Control trace-context provider after a turn."""
    if otel_sink_active():
        _ac.clear_trace_context_provider()


def setup_agent_control() -> bool:
    """Initialize Agent Control against the configured Galileo log stream.

    Resolves the log stream ID with a short-lived Galileo logger, then points the
    Agent Control SDK at it. Best-effort: any failure logs and leaves controls off
    so the app keeps serving.
    """
    global _initialized
    if not _ENABLED or _initialized:
        return _initialized

    settings = get_settings()
    if not settings.agent_control_url or not settings.galileo_api_key:
        log.warning(
            "Agent Control enabled but AGENT_CONTROL_URL / GALILEO_API_KEY is "
            "missing; controls stay off."
        )
        return False

    # The Galileo/Agent Control SDKs read auth + URLs from os.environ. Populate
    # them from settings (and derive the multitenant console URL from the OTLP
    # endpoint) so this works whether env came from the shell or pydantic's .env.
    os.environ.setdefault("GALILEO_API_KEY", settings.galileo_api_key)
    os.environ.setdefault("GALILEO_PROJECT", settings.galileo_project)
    os.environ.setdefault("GALILEO_LOG_STREAM", settings.galileo_log_stream)
    os.environ.setdefault("AGENT_CONTROL_URL", settings.agent_control_url)
    # The galileo.luna scorer (e.g. Input PII SLM) authenticates with a SECRET key.
    if settings.galileo_api_secret_key:
        os.environ.setdefault("GALILEO_API_SECRET_KEY", settings.galileo_api_secret_key)
    if settings.galileo_luna_invoke_url:
        os.environ.setdefault("GALILEO_LUNA_INVOKE_URL", settings.galileo_luna_invoke_url)
    console = settings.galileo_console_url
    if not console and settings.galileo_otel_endpoint:
        host = urlparse(settings.galileo_otel_endpoint).netloc
        if host.startswith("api."):
            console = f"https://{host.replace('api.', 'console.', 1)}"
    if console:
        os.environ.setdefault("GALILEO_CONSOLE_URL", console)
    api_url = settings.galileo_api_url
    if not api_url and settings.galileo_otel_endpoint:
        host = urlparse(settings.galileo_otel_endpoint).netloc
        if host:
            api_url = f"https://{host}"
    if api_url:
        os.environ.setdefault("GALILEO_API_URL", api_url)

    # Controls are bound to a specific log stream; target the one that holds them.
    ac_log_stream = settings.agent_control_log_stream or settings.galileo_log_stream
    try:
        from galileo.logger.logger import GalileoLogger  # noqa: PLC0415

        logger = GalileoLogger(
            project=settings.galileo_project,
            log_stream=ac_log_stream,
        )
        if logger.log_stream_id is None:
            raise RuntimeError("could not resolve Galileo log stream ID")

        # Enable the native bridge only when at least one domain traces natively.
        # The bridge attaches the rich @control span (Controls tab, evaluator
        # breakdown) to an active logger trace; OTel-domain turns have no active
        # logger trace, so their control step is the app's OTel guardrail span.
        otel_sink = settings.agent_control_otel_sink
        native = bool(settings.native_domains()) and not otel_sink
        if native:
            logger.enable_agent_control()
            logger.start_session(name=settings.agent_control_agent_name)

        init_kwargs = dict(
            agent_name=settings.agent_control_agent_name,
            agent_description="ai_agent_demo runtime agent",
            server_url=settings.agent_control_url,
            api_key=settings.galileo_api_key,
            api_key_header=settings.agent_control_api_key_header,
            observability_enabled=native or otel_sink,
            target_type=settings.agent_control_target_type,
            target_id=logger.log_stream_id,
        )
        if otel_sink:
            # Control spans over OTLP (typed "control" spans), nested into the
            # app's OTel trace via set_trace_context_provider (see main.py).
            init_kwargs["observability_sink_name"] = "otel"
            init_kwargs["observability_sink_config"] = {
                "enabled": True,
                "endpoint": settings.galileo_otel_endpoint,
                "headers": {
                    settings.agent_control_api_key_header: settings.galileo_api_key,
                    "projectid": logger.project_id,
                    "logstreamid": logger.log_stream_id,
                },
                "service_name": settings.otel_service_name,
            }
        elif native:
            init_kwargs["observability_sink_name"] = "registered"
        _ac.init(**init_kwargs)
        global _logger, _otel_sink
        _logger = logger
        _otel_sink = otel_sink
        _initialized = True
        try:
            n = len(_ac.get_server_controls())
        except Exception:  # noqa: BLE001
            n = -1
        log.info(
            "Agent Control initialised (agent=%s, log_stream=%s, log_stream_id=%s, controls=%d)",
            settings.agent_control_agent_name,
            ac_log_stream,
            logger.log_stream_id,
            n,
        )
    except Exception:  # noqa: BLE001
        log.exception("Agent Control init failed; continuing without controls.")
    return _initialized


@control(step_name="user_input")
async def evaluate_user_input(text: str) -> str:
    """Controlled step for the incoming user message (e.g. prompt-injection/PII)."""
    return text


@control(step_name="assistant_output")
async def evaluate_assistant_output(text: str) -> str:
    """Controlled step for the final assistant answer (e.g. output PII/toxicity).

    The answer is passed here as the step INPUT, so a console control that scores
    the answer must read Payload Field = input (NOT output) and target this step.
    """
    return text


def describe_control_error(exc: Exception) -> dict:
    """Pull the control's identity and its own wording out of a control exception.

    The SDK sets ``control_name``/``message`` on both error types and
    ``steering_context`` on steers; the offline fallback stubs default them.
    """
    detail = ""
    for candidate in (
        getattr(exc, "steering_context", ""),
        getattr(exc, "message", ""),
    ):
        if isinstance(candidate, dict):
            candidate = candidate.get("message")
        if not isinstance(candidate, str):
            continue
        candidate = candidate.strip()
        if candidate and candidate != "No steering context provided":
            detail = candidate
            break
    # No str(exc) fallback: its repr renders "…: None" when the server sends a null
    # message. The raw text still reaches the UI via "reason" below.
    return {
        "action": "steer" if isinstance(exc, ControlSteerError) else "deny",
        "control": str(getattr(exc, "control_name", "") or "unknown"),
        "detail": detail,
        "reason": f"Agent Control: {exc}",
    }


def control_block_message(info: dict) -> str:
    if info["detail"]:
        return (
            f"Answer withheld — Galileo Agent Control ({info['control']}): "
            f"{info['detail']}"
        )
    verb = "steered" if info["action"] == "steer" else "denied"
    return (
        f"Answer withheld — Galileo Agent Control {verb} this response "
        f"(control: {info['control']})."
    )
