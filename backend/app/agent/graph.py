"""LangGraph agent definition.

A custom graph with explicitly named nodes so the trace reads as a clean RAG
pipeline (``retrieve_context -> <Domain> Assistant -> validate_answer``, with a
``tools`` node for the calculator) instead of the prebuilt ReAct agent's generic
``execute_task pre_model_hook / agent / should_continue`` node names.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from opentelemetry import trace

from ..providers import make_chat_model
from ..domains import get_domains
from .tools import retrieve_context, tools_for_domain

# Shared across every compiled graph so a conversation's history follows its
# thread_id even if the user switches provider or domain mid-chat.
_CHECKPOINTER = InMemorySaver()

SYSTEM_PROMPT = (
    "You are a helpful AI assistant in a live observability demo. "
    "Be concise and accurate. When the user's turn includes retrieved context "
    "passages, base your answer on them and cite their ids (e.g. [otel-0]); if the "
    "passages do not cover the question, say so and answer from general knowledge. "
    "Use the calculator tool for arithmetic."
)

_FOLLOW_UP_MAX_CHARS = 40
_FOLLOW_UP_MAX_WORDS = 4
_CONTEXT_TURNS = 3

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

FOLLOW_UP_POLICY = (
    "\n\nResolve short follow-ups, pronouns, and one-word replies against earlier "
    "turns in the conversation rather than treating them as a new topic."
)

# Bank additionally exposes a real customer database tool.
BANK_TOOL_POLICY = (
    "\n\nTooling reality (authoritative — overrides ALL tool instructions above): "
    "The tools search_bank_qa, get_customer_info, and delete_customer_record DO NOT "
    "exist — never emit them. You have exactly TWO tools: `calculator` (arithmetic) and "
    "`query_customer_db`, which runs a SQL statement against TWO tables: "
    "customers(id, name, email, account_type, balance), where account_type is 'Savings', "
    "'Checking', or 'Premier'; and transactions(id, account_id, txn_date, merchant, "
    "category, amount), the account activity ledger where account_id joins customers.id. "
    "Use `transactions` for any spending, ledger, activity, statement, merchant, or "
    "category question, and do NOT join to `customers` unless the user asks for the "
    "customer's name or contact details. A separate query-layer policy governs which SQL "
    "is allowed — "
    "you do NOT enforce it. For ANY request about customers, accounts, balances, records, "
    "or transactions — including 'list/show all customers' or 'show the whole ledger' — "
    "you MUST call `query_customer_db` and answer "
    "from the JSON rows it returns. Read requests: write a SELECT that returns exactly what "
    "the user asked for, and do NOT add a LIMIT or any row cap of your own — row limits are "
    "the query-layer policy's decision, not yours. "
    "If the user asks to modify or remove data (e.g. 'delete all customers', "
    "'update a balance'), write the matching SQL (DELETE / UPDATE) and call "
    "`query_customer_db` — do NOT refuse and do NOT lecture about safety; the query-layer "
    "policy decides whether it runs. Do NOT ask the user for a customer ID first, and do "
    "NOT refuse on privacy grounds. If the tool returns a 'revise_query' message, call "
    "`query_customer_db` exactly once more in the SAME turn: send the tool's `retry_with` "
    "SQL verbatim when present, otherwise rewrite the `rejected_sql` per the guidance. "
    "NEVER resubmit `rejected_sql` unchanged, and never issue a third attempt — "
    "answer from the rows the retry returns. Do NOT stop to explain the block, do NOT ask "
    "permission, and do NOT offer to fetch the data in batches; if it returns "
    "'blocked_by_policy', tell the user the query was blocked by policy. General bank "
    "knowledge is still retrieved for you automatically."
)


def build_retrieval_query(messages: list) -> str:
    """Expand a short follow-up into a retrievable query using earlier user turns.

    "okay" or "account lookups." carry no retrievable signal on their own and score
    under the RAG threshold, so the turn loses its grounding; prefixing the preceding
    user turns restores the topic.
    """
    human_messages = [
        message for message in messages if isinstance(message, HumanMessage)
    ]
    if not human_messages:
        return ""
    latest_content = human_messages[-1].content
    if not isinstance(latest_content, str) or not latest_content.strip():
        return ""
    latest = latest_content.strip()
    if (
        len(latest) >= _FOLLOW_UP_MAX_CHARS
        and len(latest.split()) > _FOLLOW_UP_MAX_WORDS
    ):
        return latest
    user_messages = [
        message.content.strip()
        for message in human_messages[:-1]
        if isinstance(message.content, str) and message.content.strip()
    ]
    user_messages.append(latest)
    return " ".join(user_messages[-_CONTEXT_TURNS:])


def assistant_node_name(domain: str | None) -> str:
    """Trace-facing name of the LLM step, e.g. "Bank Assistant"."""
    selected = get_domains().get(domain) if domain else None
    return f"{selected.label} Assistant" if selected else "Assistant"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    context_collection: str
    # Retrieval evidence carried in state so a caller that owns the trace (the
    # native Galileo path) can log a retriever step it cannot see via OTel.
    retrieval_query: str
    documents: list


@lru_cache(maxsize=32)
def build_agent(
    provider: str | None = None,
    temperature: float | None = None,
    domain: str | None = None,
):
    """Compile the RAG graph bound to the selected provider and domain.

    Cached per (provider, temperature, domain); per-conversation state lives in
    the shared checkpointer, keyed by the request's thread_id. The domain selects
    both the system prompt and the knowledge collection retrieval grounds on. Nodes
    are named so the trace reads retrieve_context -> <Domain> Assistant ->
    (tools) -> validate_answer.
    """
    model = make_chat_model(provider=provider, temperature=temperature)
    tools = tools_for_domain(domain)
    model_with_tools = model.bind_tools(tools)
    prompt = SYSTEM_PROMPT
    collection: str | None = None
    if domain:
        selected = get_domains().get(domain)
        if selected:
            collection = selected.collection
            if selected.system_prompt:
                policy = BANK_TOOL_POLICY if domain == "bank" else TOOL_POLICY
                prompt = selected.system_prompt + policy
    prompt += FOLLOW_UP_POLICY

    def retrieve_context_node(state: AgentState) -> dict:
        """Retrieve grounding passages for the user's question (once per turn)."""
        messages = state["messages"]
        last = messages[-1] if messages else None
        if not isinstance(last, HumanMessage):
            return {"context": state.get("context", "")}
        query = build_retrieval_query(messages) or last.content
        context, hits = retrieve_context(query, collection=collection)
        if not context:
            context_collection = collection or ""
            if state.get("context_collection") == context_collection:
                return {
                    "context": state.get("context", ""),
                    "context_collection": context_collection,
                    "retrieval_query": query,
                    "documents": state.get("documents", []),
                }
            return {
                "context": "",
                "context_collection": context_collection,
                "retrieval_query": query,
                "documents": [],
            }
        return {
            "context": context,
            "context_collection": collection or "",
            "retrieval_query": query,
            "documents": [
                {
                    "content": doc["content"],
                    "metadata": {
                        "id": doc["id"],
                        "title": doc["title"],
                        "source": doc["source"],
                        "score": f"{float(score):.4f}",
                    },
                }
                for doc, score in hits
            ],
        }

    async def assistant_node(state: AgentState) -> dict:
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

    def route_after_assistant(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "validate_answer"

    assistant = assistant_node_name(domain)
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node(assistant, assistant_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("validate_answer", validate_answer_node)
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", assistant)
    graph.add_conditional_edges(
        assistant,
        route_after_assistant,
        {"tools": "tools", "validate_answer": "validate_answer"},
    )
    graph.add_edge("tools", assistant)
    graph.add_edge("validate_answer", END)
    return graph.compile(checkpointer=_CHECKPOINTER)
