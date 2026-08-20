"""Minimal Galileo experiment on the OpenTelemetry path.

Follows the official guide:
https://docs.galileo.ai/how-to-guides/experiments/otel-experiment/otel-experiment

The experiment runner calls ``run_agent`` per dataset row; the agent emits OTel
spans that a ``GalileoSpanProcessor`` routes to the experiment. Auth comes from
.env (GALILEO_API_KEY, GALILEO_PROJECT); nothing is hardcoded.
"""

import os

# Disable the native Galileo logger BEFORE importing galileo/agent code, so the
# OTel spans are the single source of truth (per the guide).
os.environ["GALILEO_LOGGING_DISABLED"] = "true"

import ast
import asyncio
import operator
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()
# The GalileoSpanProcessor derives its OTLP endpoint from the console URL.
if not os.environ.get("GALILEO_CONSOLE_URL"):
    host = urlparse(os.environ.get("GALILEO_OTEL_ENDPOINT", "")).netloc
    if host.startswith("api."):
        os.environ["GALILEO_CONSOLE_URL"] = f"https://{host.replace('api.', 'console.', 1)}"

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from opentelemetry import trace
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.trace import TracerProvider

from galileo import GalileoMetrics
from galileo.experiments import run_experiment
from galileo.otel import GalileoSpanProcessor, add_galileo_span_processor

PROJECT = os.environ["GALILEO_PROJECT"]

# ── OTel setup (per the guide) ───────────────────────────────────────────────
tracer_provider = TracerProvider()
add_galileo_span_processor(tracer_provider, GalileoSpanProcessor(project=PROJECT))
trace.set_tracer_provider(tracer_provider)
# Framework instrumentation: emits nested LLM/tool spans (the guide uses each
# framework's own enable_instrumentation; for LangChain it's this instrumentor).
LangchainInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = trace.get_tracer("miniexp")

# ── Agent ────────────────────────────────────────────────────────────────────
MODEL = os.environ.get("LOCAL_MODEL")
llm = ChatOpenAI(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
    model=MODEL,
    temperature=0,
)
SYSTEM = (
    "You are a concise, helpful assistant. For any Singapore (SG) local topic "
    "such as weather or haze, call sg_info_center and cite the official source. "
    "For arithmetic, call calculator. If you lack real-time data, say so."
)


# ── Tools (a calculator + a small SG info-center RAG lookup) ──────────────────
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '17 * 23 + 4'."""
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


# A tiny curated knowledge base of authoritative Singapore sources.
_SG_SOURCES = {
    "weather": "NEA (nea.gov.sg) and the myENV app — 2-hour nowcast, 4-day outlook, and rain-area radar.",
    "haze": "NEA PSI readings at nea.gov.sg/psi and the haze microsite.",
    "transport": "LTA (lta.gov.sg) and the MyTransport.SG app for MRT/bus arrivals.",
    "dengue": "NEA dengue clusters at nea.gov.sg/dengue.",
}


@tool
def sg_info_center(topic: str) -> str:
    """Authoritative Singapore source(s) for a local topic such as weather, haze,
    transport, or dengue. Use this for any Singapore/SG question."""
    key = topic.lower()
    for name, source in _SG_SOURCES.items():
        if name in key:
            return source
    return "For official Singapore info start at gov.sg and the relevant agency (NEA, LTA, MOH)."


# ReAct agent so the model can call the tools; LangchainInstrumentor traces the
# agent, tool, and LLM spans under our root span.
agent = create_react_agent(llm, tools=[calculator, sg_info_center], prompt=SYSTEM)


async def _run_agent_async(user_message: str) -> str:
    # Galileo only accepts traces with a valid GenAI root span (frameworks like
    # the guide's emit one automatically; here we set one around the agent run).
    with tracer.start_as_current_span("invoke_agent mini") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        span.set_attribute("gen_ai.provider.name", "ollama")
        span.set_attribute("gen_ai.request.model", MODEL)
        span.set_attribute("gen_ai.agent.name", "mini")
        span.set_attribute("input.value", user_message)
        result = await agent.ainvoke({"messages": [("user", user_message)]})
        answer = result["messages"][-1].content or ""
        span.set_attribute("output.value", answer)
        return answer


def run_agent(input_data):
    """Called by run_experiment for each dataset row (input is str or dict)."""
    if isinstance(input_data, str):
        user_message = input_data
    elif isinstance(input_data, dict):
        user_message = input_data.get("input", "")
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)!r}")
    return asyncio.run(_run_agent_async(user_message))


# ── Run the experiment (per the guide) ───────────────────────────────────────
DATASET = [
    {
        "input": "what is the weather today in SG?",
        "ground_truth": (
            "This AI agent is not connected to the internet, please check NEA's "
            "website for the latest weather information."
        ),
    },
    {
        "input": "what is 17 * 23 + 4?",
        "ground_truth": "395",
    },
]

result = run_experiment(
    "mini-otel-experiment",
    dataset=DATASET,
    function=run_agent,
    metrics=[GalileoMetrics.correctness, GalileoMetrics.completeness_luna, GalileoMetrics.context_adherence_luna],
    project=PROJECT,
)
# Export any spans still buffered before the process exits.
tracer_provider.force_flush()
print("done:", result.get("link") if isinstance(result, dict) else result)
