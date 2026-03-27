"""Best-effort PII redaction for free-text (e.g. chat) before logging or LLM use."""

import re
from typing import Tuple

# US SSN (with common separators)
_SSN = re.compile(
    r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    re.IGNORECASE,
)
# 16-digit card (simple)
_CARD = re.compile(r"\b(?:\d[ -]*?){15,16}\d\b")
# Email
_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
)
# US phone
_PHONE = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
)


def redact_pii(text: str) -> Tuple[str, int]:
    """Redact common PII patterns. Returns (redacted_text, number_of_replacements)."""
    if not text:
        return text, 0
    count = 0
    s = text

    def _sub(pattern, repl, string):
        nonlocal count
        new, n = pattern.subn(repl, string)
        count += n
        return new

    s = _sub(_SSN, "[REDACTED-SSN]", s)
    s = _sub(_CARD, "[REDACTED-PAYMENT]", s)
    s = _sub(_EMAIL, "[REDACTED-EMAIL]", s)
    s = _sub(_PHONE, "[REDACTED-PHONE]", s)
    return s, count
