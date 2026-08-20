"""Pluggable guardrail interface with a lightweight, dependency-free default.

The interface is deliberately generic. Swap ``get_guardrail()`` to return a
Galileo Protect, NeMo Guardrails, or Presidio-backed implementation without
touching the agent or the API layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ..privacy import redact_pii


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""
    redacted_text: str | None = None


class Guardrail(Protocol):
    def check_input(self, text: str) -> GuardrailResult: ...
    def check_output(self, text: str) -> GuardrailResult: ...


class DefaultGuardrail:
    """PII redaction on output + naive jailbreak/prompt-injection block on input."""

    _JAILBREAK = re.compile(
        r"ignore\s+(?:all\s+|any\s+|the\s+)*(?:previous|prior|earlier|preceding|above)?"
        r"\s*instructions|jailbreak|DAN mode|do anything now",
        re.IGNORECASE,
    )

    def check_input(self, text: str) -> GuardrailResult:
        if self._JAILBREAK.search(text):
            return GuardrailResult(
                allowed=False, reason="possible prompt-injection / jailbreak attempt"
            )
        return GuardrailResult(allowed=True)

    def check_output(self, text: str) -> GuardrailResult:
        return GuardrailResult(allowed=True, redacted_text=redact_pii(text))


def get_guardrail() -> Guardrail:
    """Return the active guardrail implementation (swap vendors here)."""
    return DefaultGuardrail()
