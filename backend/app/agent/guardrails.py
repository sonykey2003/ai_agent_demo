"""Pluggable guardrail interface with a lightweight, dependency-free default.

The interface is deliberately generic. Swap ``get_guardrail()`` to return a
Galileo Protect, NeMo Guardrails, or Presidio-backed implementation without
touching the agent or the API layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


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

    _EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    _PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b")
    _SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
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
        redacted = self._SSN.sub("[REDACTED-SSN]", text)
        redacted = self._EMAIL.sub("[REDACTED-EMAIL]", redacted)
        redacted = self._PHONE.sub("[REDACTED-PHONE]", redacted)
        return GuardrailResult(allowed=True, redacted_text=redacted)


def get_guardrail() -> Guardrail:
    """Return the active guardrail implementation (swap vendors here)."""
    return DefaultGuardrail()
