"""Intent classification.

The goal of this module is to turn an unstructured ``CompletionRequest``
into an :class:`Intent` that the routing policy can consume.

We deliberately keep this *simple and deterministic* rather than calling
an ML model: the distributed-systems focus of the project is not in the
classifier, and a heuristic classifier is easy to unit test.
"""
from __future__ import annotations

import re

from ..common.models import ChatMessage, CompletionRequest, Intent


_CODE_HINTS = (
    "```", "def ", "class ", "function ", "import ", "const ", "let ",
    "<html", "SELECT ", "select ", "stack trace", "traceback",
)
_REASONING_HINTS = (
    "explain", "reason", "prove", "derive", "analyze", "analyse", "compare",
    "step by step", "why ", "how does", "trade-off", "tradeoff",
)
_BATCH_HINTS = (
    "summarize the following", "translate the following", "extract",
    "generate a list of", "batch",
)


def _joined_text(messages: list[ChatMessage]) -> str:
    return "\n".join(m.content for m in messages).lower()


def classify(request: CompletionRequest) -> Intent:
    """Return the inferred intent for ``request``.

    Priority order:

    1. An explicit ``intent_hint`` always wins.
    2. Long prompts with reasoning keywords are REASONING.
    3. Presence of code hints flags CODE.
    4. Explicit batch markers flag BATCH.
    5. Short prompts default to INTERACTIVE.
    """
    if request.intent_hint is not None:
        return request.intent_hint

    text = _joined_text(request.messages)
    total_chars = len(text)

    # Batch first: an explicit batch prefix should outrank anything else.
    if any(h in text for h in _BATCH_HINTS):
        return Intent.BATCH

    if any(h in text for h in _CODE_HINTS) or re.search(r"\b(bug|error|exception)\b", text):
        return Intent.CODE

    reasoning_hits = sum(1 for h in _REASONING_HINTS if h in text)
    if reasoning_hits >= 1 and total_chars > 200:
        return Intent.REASONING
    if reasoning_hits >= 2:
        return Intent.REASONING

    return Intent.INTERACTIVE
