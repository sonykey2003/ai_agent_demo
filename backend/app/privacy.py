"""Shared privacy helpers for user-visible output and exported telemetry."""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def redact_pii(text: str) -> str:
    """Mask known PII patterns while preserving the surrounding content."""
    redacted = _SSN.sub("[REDACTED-SSN]", text)
    redacted = _EMAIL.sub("[REDACTED-EMAIL]", redacted)
    return _PHONE.sub("[REDACTED-PHONE]", redacted)
