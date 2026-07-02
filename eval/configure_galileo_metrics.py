"""One-time Galileo setup: register RAG-quality judge metrics scored by your own
NVIDIA NIM model (instead of Galileo's default ``gpt-4.1-mini`` scorer).

Why this exists
---------------
Galileo's out-of-the-box RAG metrics (Chunk Attribution/Utilization, Context
Adherence, Completeness) are hard-pinned to the ``gpt-4.1-mini`` scorer, which
requires an OpenAI integration. There is no per-metric model picker for them, so
connecting NVIDIA NIM or Anthropic does not make them work ("model with alias
gpt-4.1-mini not found").

Custom LLM-as-judge metrics *do* let you choose the scorer model. This script
registers two of them - Context Adherence and Completeness - pointed at a model
your Galileo project already integrates (your NVIDIA NIM model), so the demo needs
no OpenAI key.

This is a Galileo-specific *admin helper*, intentionally separate from the
vendor-neutral application. The app never imports the Galileo SDK.

Usage
-----
    pip install -r eval/requirements-galileo.txt

    # In .env (or your shell), make sure these are set:
    #   GALILEO_API_KEY=...                 (Settings -> API Keys)
    #   GALILEO_CONSOLE_URL=https://console.multitenant.galileocloud.io
    #   GALILEO_JUDGE_MODEL=moonshotai/kimi-k2.6   # alias Galileo lists for your NIM model
    python eval/configure_galileo_metrics.py

Then open your log stream -> Configure Metrics, toggle the two new metrics ON, and
re-run a knowledge question (e.g. "What is Galileo and how does it relate to
OpenTelemetry?").
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

CONTEXT_ADHERENCE_PROMPT = """\
You are an impartial evaluator measuring whether an AI assistant's response is
grounded in the information that was available to it.

You are given the assistant's input - which may include retrieved context passages
and the user's question - and the assistant's output response.

Task: Decide whether every factual claim in the output is supported by the
information present in the input (the retrieved context and the question).

Set true if the response is fully grounded: every claim is supported by the
provided context, with no fabricated or unsupported information.
Set false if the response contains any claim that the provided context does not
support.
"""

COMPLETENESS_PROMPT = """\
You are an impartial evaluator measuring how completely an AI assistant's response
answers the user's question using the information available to it.

You are given the assistant's input - which may include retrieved context passages
and the user's question - and the assistant's output response.

Task: Judge what fraction of the user's question is fully and accurately addressed
by the response, using the information available in the provided context.

Return 1.0 when the response completely and accurately addresses every part of the
question. Return proportionally lower values toward 0.0 when parts of the question
are unanswered, only partially answered, or inaccurate.
"""


def _resolve_console_url() -> str | None:
    """Use GALILEO_CONSOLE_URL, else derive it from the OTLP ingest endpoint."""
    explicit = os.environ.get("GALILEO_CONSOLE_URL")
    if explicit:
        return explicit
    otel = os.environ.get("GALILEO_OTEL_ENDPOINT", "")
    host = urlparse(otel).netloc
    if host.startswith("api."):
        return f"https://{host.replace('api.', 'console.', 1)}"
    return None


def main() -> int:
    if not os.environ.get("GALILEO_API_KEY"):
        print("ERROR: GALILEO_API_KEY is not set (see .env).")
        return 1

    console_url = _resolve_console_url()
    if console_url:
        os.environ.setdefault("GALILEO_CONSOLE_URL", console_url)
    if not os.environ.get("GALILEO_CONSOLE_URL"):
        print(
            "ERROR: could not determine the Galileo console URL. Set "
            "GALILEO_CONSOLE_URL, e.g. https://console.multitenant.galileocloud.io"
        )
        return 1

    judge_model = os.environ.get("GALILEO_JUDGE_MODEL", "moonshotai/kimi-k2.6")

    try:
        from galileo.metrics import (  # noqa: PLC0415
            OutputTypeEnum,
            StepType,
            create_custom_llm_metric,
        )
    except ImportError:
        print("ERROR: Galileo SDK missing. Run: pip install -r eval/requirements-galileo.txt")
        return 1

    print(f"Console : {os.environ['GALILEO_CONSOLE_URL']}")
    print(f"Judge   : {judge_model}")

    metrics = [
        {
            "name": "Context Adherence - NIM judge",
            "user_prompt": CONTEXT_ADHERENCE_PROMPT,
            "output_type": OutputTypeEnum.BOOLEAN,
            "description": (
                "Is the assistant's answer grounded in the retrieved context? "
                "Judged by your NVIDIA NIM model."
            ),
        },
        {
            "name": "Completeness - NIM judge",
            "user_prompt": COMPLETENESS_PROMPT,
            "output_type": OutputTypeEnum.PERCENTAGE,
            "description": (
                "How fully does the answer address the question using the retrieved "
                "context? Judged by your NVIDIA NIM model."
            ),
        },
    ]

    for metric in metrics:
        try:
            create_custom_llm_metric(
                name=metric["name"],
                user_prompt=metric["user_prompt"],
                # Evaluate each LLM call: its input carries the injected context +
                # question, its output carries the grounded answer.
                node_level=StepType.llm,
                cot_enabled=True,
                model_name=judge_model,
                num_judges=1,
                description=metric["description"],
                tags=["rag", "nim-judge"],
                output_type=metric["output_type"],
            )
            print(f"  created: {metric['name']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED  {metric['name']}: {exc}")
            if "not found" in str(exc).lower() or "model" in str(exc).lower():
                print(
                    "    -> Set GALILEO_JUDGE_MODEL to the EXACT model alias listed "
                    "under Settings -> Integrations -> NVIDIA for your project."
                )
            return 1

    print(
        "\nDone. Now open your log stream -> Configure Metrics, toggle ON:\n"
        "  - Context Adherence - NIM judge\n"
        "  - Completeness - NIM judge\n"
        "then re-run a knowledge question to see them populate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
