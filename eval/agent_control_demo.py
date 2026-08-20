"""Galileo Agent Control demo (native GalileoLogger path).

Follows: https://docs.galileo.ai/how-to-guides/agent-control/initialize-and-configure-agent-control

Agent Control is NOT part of the vendor-neutral runtime app (the backend never
imports the Galileo SDK). This is a standalone admin/demo helper, like the other
``eval/`` scripts. It initializes the Galileo Logger + the Agent Control SDK,
runs a small agent whose LLM/tool steps are wrapped with ``@control()``, and
bridges the resulting control spans back into Galileo logging.

Prerequisites (this script cannot run without them):
  1. Install the Agent Control SDK. It is distributed with your Galileo
     deployment and is NOT on public PyPI, so install it from your Galileo
     distribution / index, e.g.:
         pip install agent-control agent-control-telemetry
  2. Create a control in the Galileo console, bound to the same project and log
     stream used below:
         https://docs.galileo.ai/how-to-guides/agent-control/create-a-control
  3. Set these in .env / the environment:
         GALILEO_API_KEY, GALILEO_PROJECT, GALILEO_LOG_STREAM   (already in .env)
         AGENT_CONTROL_URL=https://agent-control.<env>.galileocloud.io
         # optional:
         AGENT_CONTROL_AGENT_NAME=ai-agent-demo
         AGENT_CONTROL_API_KEY_HEADER=Galileo-API-Key
         GALILEO_LOGGER_MODE=batch

Usage:
    pip install -r eval/requirements-galileo.txt   # + the Agent Control SDK (above)
    python eval/agent_control_demo.py
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _resolve_console_url() -> None:
    if os.environ.get("GALILEO_CONSOLE_URL"):
        return
    host = urlparse(os.environ.get("GALILEO_OTEL_ENDPOINT", "")).netloc
    if host.startswith("api."):
        os.environ["GALILEO_CONSOLE_URL"] = f"https://{host.replace('api.', 'console.', 1)}"


SYSTEM = (
    "You are a cautious banking operations assistant. Summarise the requested "
    "action in one sentence and note any risk. Do not actually move money."
)


def main() -> int:
    _load_dotenv()
    _resolve_console_url()

    # 1. The Agent Control SDK is required (enterprise; not on public PyPI).
    try:
        import agent_control
        from agent_control import control
    except ImportError:
        print(
            "ERROR: Agent Control SDK not installed.\n"
            "  Install it from your Galileo distribution, e.g.:\n"
            "    pip install agent-control agent-control-telemetry\n"
            "  (it is not on public PyPI). See "
            "https://docs.galileo.ai/concepts/agent-control/overview"
        )
        return 1

    api_key = os.environ.get("GALILEO_API_KEY")
    server_url = os.environ.get("AGENT_CONTROL_URL")
    if not api_key:
        print("ERROR: GALILEO_API_KEY is not set (see .env).")
        return 1
    if not server_url:
        print(
            "ERROR: AGENT_CONTROL_URL is not set. Point it at your Agent Control "
            "server, e.g. https://agent-control.multitenant.galileocloud.io"
        )
        return 1

    from galileo import get_agent_control_target
    from galileo.logger.logger import GalileoLogger
    from langchain_openai import ChatOpenAI

    project = os.environ["GALILEO_PROJECT"]
    log_stream = os.environ["GALILEO_LOG_STREAM"]

    # 2. Initialize the Galileo Logger and open a session for this agent run.
    logger = GalileoLogger(
        project=project,
        log_stream=log_stream,
        mode=os.environ.get("GALILEO_LOGGER_MODE", "batch"),
    )
    logger.start_session(
        name=os.environ.get("GALILEO_SESSION_NAME", "agent-control-demo"),
        external_id=f"agent-control-demo-{uuid4()}",
        metadata={"source": "agent-control-demo"},
    )
    if logger.project_id is None or logger.log_stream_id is None:
        raise RuntimeError("Galileo logger did not resolve project/Log stream IDs.")
    os.environ["GALILEO_PROJECT_ID"] = logger.project_id
    os.environ["GALILEO_LOG_STREAM_ID"] = logger.log_stream_id

    # 3. Initialize Agent Control against the resolved log stream, and bridge its
    #    control spans back into the Galileo logger hierarchy.
    target = get_agent_control_target(log_stream_id=logger.log_stream_id)
    agent_control.init(
        agent_name=os.environ.get("AGENT_CONTROL_AGENT_NAME", "ai-agent-demo"),
        agent_description="ai_agent_demo Agent Control demo",
        server_url=server_url,
        api_key=api_key,
        api_key_header=os.environ.get("AGENT_CONTROL_API_KEY_HEADER", "Galileo-API-Key"),
        observability_enabled=True,
        observability_sink_name="registered",
        target_type=target.target_type,
        target_id=target.target_id,
    )
    logger.enable_agent_control()

    # 4. Decorate the LLM and tool steps so controls can evaluate them.
    llm = ChatOpenAI(
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
        model=os.environ.get("LOCAL_MODEL", "qwen2.5:0.5b"),
        temperature=0,
    )

    @control()
    def call_llm(message: str) -> str:
        return llm.invoke([("system", SYSTEM), ("user", message)]).content or ""

    @control()
    def transfer_funds(amount: float, recipient: str) -> str:
        # Guarded by the control; this demo never actually moves money.
        return f"[demo] would queue a transfer of ${amount:,.2f} to {recipient}."

    user_request = "Wire $15,000 to Horizon Robotics."
    trace = logger.start_trace(name="agent-control-demo-run", input={"user_request": user_request})
    decision = result = None
    try:
        decision = call_llm(user_request)
        result = transfer_funds(15000, "Horizon Robotics")
    finally:
        logger.conclude(output={"decision": decision, "result": result})
        logger.flush()

    console = os.environ.get("GALILEO_CONSOLE_URL", "").rstrip("/")
    print(f"done: control spans flushed to project '{project}' / log stream '{log_stream}'.")
    if console:
        print(f"View the Controls / Traces tab: {console}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
