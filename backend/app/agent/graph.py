"""LangGraph agent definition.

A prebuilt ReAct agent keeps the demo robust across very different model sizes:
capable cloud models (OpenAI / NIM-DeepSeek) will call tools, while a tiny local
model can still answer directly. Either way the graph emits clean spans.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from ..providers import make_chat_model
from .tools import TOOLS, retrieve_context

SYSTEM_PROMPT = (
    "You are a helpful AI assistant in a live observability demo. "
    "Be concise and accurate. When the user's turn includes retrieved context "
    "passages, base your answer on them and cite their ids (e.g. [kb-otel]); if the "
    "passages do not cover the question, say so and answer from general knowledge. "
    "Use the calculator tool for arithmetic."
)


def _pre_model_hook(state):
    """Retrieve-then-generate, executed inside the agent graph.

    Running retrieval here (rather than around the agent) keeps the retriever span
    nested under the agent trace, so the whole turn is one trace rooted at
    ``invoke_agent`` and RAG evaluators can line up context with the response.
    Only fires on the first model call (when the latest message is the user's).
    """
    messages = state["messages"]
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {"llm_input_messages": messages}
    context, _hits = retrieve_context(messages[-1].content)
    if not context:
        return {"llm_input_messages": messages}
    return {"llm_input_messages": [SystemMessage(content=f"Relevant context passages:\n\n{context}"), *messages]}


@lru_cache(maxsize=16)
def build_agent(provider: str | None = None, temperature: float | None = None):
    """Construct a ReAct agent bound to the selected provider.

    Cached per (provider, temperature): the compiled graph is stateless and
    reusable across requests, so the agent is built once instead of on every
    chat. This also keeps the one-time "create_agent" construction span out of
    each request's trace.
    """
    model = make_chat_model(provider=provider, temperature=temperature)
    return create_react_agent(model, TOOLS, prompt=SYSTEM_PROMPT, pre_model_hook=_pre_model_hook)
