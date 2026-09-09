import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).parents[1]))

from app import main
from app.agent.agent_control import (
    ControlSteerError,
    ControlViolationError,
    control_block_message,
    describe_control_error,
)
from app.agent.graph import _CONTEXT_TURNS, build_retrieval_query
from app.main import ChatRequest


def test_long_and_wordy_retrieval_query_is_unchanged() -> None:
    question = "Explain how customer account lookups work in the banking system."

    query = build_retrieval_query(
        [HumanMessage(content="earlier unrelated topic"), HumanMessage(content=question)]
    )

    assert query == question


def test_short_follow_up_includes_preceding_user_turn() -> None:
    question = "Explain how customer account lookups work in the banking system."

    query = build_retrieval_query(
        [HumanMessage(content=question), HumanMessage(content="okay")]
    )

    assert question in query
    assert "okay" in query


def test_five_word_short_follow_up_includes_preceding_user_turn() -> None:
    question = "Explain the available account types."
    follow_up = "what about the second one?"

    query = build_retrieval_query(
        [HumanMessage(content=question), HumanMessage(content=follow_up)]
    )

    assert question in query
    assert follow_up in query


def test_short_follow_up_uses_only_recent_user_turns() -> None:
    turns = [
        "oldest topic",
        "older detail",
        "recent topic",
        "newer detail",
        "okay",
    ]

    query = build_retrieval_query(
        [HumanMessage(content=turn) for turn in turns]
    )

    assert query == " ".join(turns[-_CONTEXT_TURNS:])
    assert turns[0] not in query


def test_blank_latest_human_message_is_empty() -> None:
    messages = [
        HumanMessage(content="Explain customer account lookups."),
        HumanMessage(content="   "),
    ]

    assert build_retrieval_query(messages) == ""


def test_retrieval_query_without_human_messages_is_empty() -> None:
    assert build_retrieval_query([]) == ""
    assert build_retrieval_query([AIMessage(content="assistant reply")]) == ""


def test_retrieval_query_excludes_ai_messages() -> None:
    question = "Explain how customer account lookups work in the banking system."
    query = build_retrieval_query(
        [
            HumanMessage(content=question),
            AIMessage(content="This text must not be retrieved."),
            HumanMessage(content="okay"),
        ]
    )

    assert question in query
    assert "okay" in query
    assert "This text must not be retrieved." not in query


def test_control_error_description_distinguishes_steer_and_deny() -> None:
    steer = ControlSteerError(
        control_name="Output policy",
        message="Control matched.",
        steering_context="Please revise the response.",
    )
    deny = ControlViolationError(
        control_name="Prompt Injection",
        message="Luna score exceeded the threshold.",
    )

    steer_info = describe_control_error(steer)
    deny_info = describe_control_error(deny)

    assert steer_info["action"] == "steer"
    assert steer_info["control"] == "Output policy"
    assert steer_info["detail"] == "Please revise the response."
    assert deny_info["action"] == "deny"
    assert deny_info["control"] == "Prompt Injection"
    assert "Output policy" in control_block_message(steer_info)
    assert "Prompt Injection" in control_block_message(deny_info)
    assert "PII" not in control_block_message(steer_info)
    assert "PII" not in control_block_message(deny_info)


def test_control_error_description_rejects_non_string_details() -> None:
    null_message = ControlViolationError(
        control_name="Null message",
        message=None,
    )
    stored_context = ControlSteerError(
        control_name="Stored context",
        message="Control matched.",
        steering_context={"message": "Please revise the response."},
    )

    for info in (
        describe_control_error(null_message),
        describe_control_error(stored_context),
    ):
        assert isinstance(info["detail"], str)
        assert info["detail"] != "None"
        assert not info["detail"].startswith("{")
        assert "None" not in control_block_message(info)
        assert info["control"] in control_block_message(info)

    # An unusable message yields no detail at all rather than the SDK's repr.
    assert describe_control_error(null_message)["detail"] == ""
    assert describe_control_error(stored_context)["detail"] == (
        "Please revise the response."
    )


def _resolve_output(
    answer: str,
    monkeypatch,
    *,
    checks,
    revision: str = "Revised text.",
    model_factory=None,
) -> tuple:
    """Drive _resolve_output_control with a scripted control + a stub chat model."""
    calls = iter(checks)

    async def fake_check(text: str) -> str:
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome
        return text

    class _StubModel:
        async def ainvoke(self, messages):
            return AIMessage(content=revision)

    monkeypatch.setattr(main, "evaluate_assistant_output", fake_check)
    monkeypatch.setattr(
        main, "make_chat_model", model_factory or (lambda **kwargs: _StubModel())
    )
    return asyncio.run(
        main._resolve_output_control(answer, ChatRequest(message="hi"))
    )


def test_clean_output_passes_the_control_untouched(monkeypatch) -> None:
    text, info = _resolve_output("All good.", monkeypatch, checks=[None])

    assert text == "All good."
    assert info is None


def test_steer_revises_the_answer_instead_of_withholding_it(monkeypatch) -> None:
    steer = ControlSteerError(
        control_name="shawn_no_PII_steer",
        message="Control triggered.",
        steering_context="The result contains PII; mask it.",
    )
    text, info = _resolve_output("Call Ada on 555-0100.", monkeypatch, checks=[steer, None])

    assert text == "Revised text."
    assert info["action"] == "steer"
    assert info["revised"] is True


def test_steer_withholds_when_the_revision_still_trips_a_control(monkeypatch) -> None:
    steer = ControlSteerError(
        control_name="shawn_no_PII_steer",
        message="Control triggered.",
        steering_context="The result contains PII; mask it.",
    )
    text, info = _resolve_output(
        "Call Ada on 555-0100.", monkeypatch, checks=[steer, steer]
    )

    assert text == control_block_message(info)
    assert info["revised"] is False


def test_deny_withholds_without_calling_the_model(monkeypatch) -> None:
    deny = ControlViolationError(
        control_name="Prompt_Injection_clone",
        message="Luna score 0.53 gte threshold 0.5",
    )

    def _no_model(**kwargs):
        raise AssertionError("deny must not attempt a revision")

    text, info = _resolve_output(
        "Anything.", monkeypatch, checks=[deny], model_factory=_no_model
    )

    assert info["action"] == "deny"
    assert info["revised"] is False
    assert text == control_block_message(info)
    assert "PII" not in text


def test_control_error_description_skips_default_steering_sentinel() -> None:
    steer = ControlSteerError(
        control_name="Output policy",
        message="Use the control message.",
    )

    assert describe_control_error(steer)["detail"] == "Use the control message."


def test_control_block_message_uses_action_verb_without_detail() -> None:
    message = control_block_message(
        {"action": "steer", "control": "Output policy", "detail": ""}
    )

    assert "steered this response" in message