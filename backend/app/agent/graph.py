"""LangGraph agent definition.

A custom graph with explicitly named nodes so the trace reads as a clean RAG
pipeline (``retrieve_context -> synthesize_answer -> validate_answer``, with a
``tools`` node for the calculator) instead of the prebuilt ReAct agent's generic
``execute_task pre_model_hook / agent / should_continue`` node names.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from opentelemetry import trace

from ..providers import make_chat_model
from ..domains import get_domains
from .tools import TOOLS, retrieve_context

SYSTEM_PROMPT = (
    "You are a helpful AI assistant in a live observability demo. "
    "Be concise and accurate. When the user's turn includes retrieved context "
    "passages, base your answer on them and cite their ids (e.g. [otel-0]); if the "
    "passages do not cover the question, say so and answer from general knowledge. "
    "Use the calculator tool for arithmetic."
)

# The reused domain prompts (golden demo) instruct the model to call tools like
# search_bank_qa / get_customer_info that this app does NOT register — only
# `calculator` exists, and knowledge is retrieved automatically. Capable models
# obey those prompts and loop on invalid tool calls, so this authoritative note is
# appended to every domain prompt to override them.
TOOL_POLICY = (
    "\n\nTooling reality (authoritative — overrides any tool instructions above): "
    "You do NOT have search_bank_qa, get_customer_info, delete_customer_record, or any "
    "database, lookup, or record-management tools. Relevant knowledge-base passages are "
    "retrieved for you automatically and included in this turn's context — answer from "
    "them and cite their ids. The ONLY callable tool is `calculator`, for arithmetic. "
    "Never emit a call to any other tool name; answer directly from the provided context "
    "instead."
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str


@lru_cache(maxsize=32)
def build_agent(
    provider: str | None = None,
    temperature: float | None = None,
    domain: str | None = None,
):
    """Compile the RAG graph bound to the selected provider and domain.

    Cached per (provider, temperature, domain): the compiled graph is stateless
    and reusable across requests. The domain selects both the system prompt and
    the knowledge collection retrieval grounds on. Nodes are named so the trace
    reads retrieve_context -> synthesize_answer -> (tools) -> validate_answer.
    """
    model = make_chat_model(provider=provider, temperature=temperature)
    model_with_tools = model.bind_tools(TOOLS)
    prompt = SYSTEM_PROMPT
    collection: str | None = None
    if domain:
        selected = get_domains().get(domain)
        if selected:
            collection = selected.collection
            if selected.system_prompt:
                prompt = selected.system_prompt + TOOL_POLICY

    def retrieve_context_node(state: AgentState) -> dict:
        """Retrieve grounding passages for the user's question (once per turn)."""
        messages = state["messages"]
        last = messages[-1] if messages else None
        if not isinstance(last, HumanMessage):
            return {"context": state.get("context", "")}
        context, _hits = retrieve_context(last.content, collection=collection)
        return {"context": context}

    async def synthesize_answer_node(state: AgentState) -> dict:
        """Generate the answer from the retrieved context; may request a tool."""
        context = state.get("context", "")
        system = (
            f"{prompt}\n\nRelevant context passages:\n\n{context}" if context else prompt
        )
        response = await model_with_tools.ainvoke(
            [SystemMessage(content=system), *state["messages"]]
        )
        return {"messages": [response]}

    def validate_answer_node(state: AgentState) -> dict:
        """Lightweight post-check recorded on the span (non-empty, cites sources)."""
        text = getattr(state["messages"][-1], "content", "") or ""
        span = trace.get_current_span()
        span.set_attribute("validation.non_empty", bool(text.strip()))
        span.set_attribute("validation.has_citation", "[" in text and "]" in text)
        return {}

    def route_after_synthesize(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "validate_answer"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("synthesize_answer", synthesize_answer_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("validate_answer", validate_answer_node)
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "synthesize_answer")
    graph.add_conditional_edges(
        "synthesize_answer",
        route_after_synthesize,
        {"tools": "tools", "validate_answer": "validate_answer"},
    )
    graph.add_edge("tools", "synthesize_answer")
    graph.add_edge("validate_answer", END)
    return graph.compile()
