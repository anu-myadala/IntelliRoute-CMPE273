"""Unit tests for the intent classifier."""
from __future__ import annotations

from intelliroute.common.models import ChatMessage, CompletionRequest, Intent
from intelliroute.router.intent import classify


def _req(text: str, **kwargs) -> CompletionRequest:
    return CompletionRequest(
        tenant_id="t1",
        messages=[ChatMessage(role="user", content=text)],
        **kwargs,
    )


def test_explicit_hint_wins():
    req = _req("anything", intent_hint=Intent.BATCH)
    assert classify(req) == Intent.BATCH


def test_short_small_talk_is_interactive():
    assert classify(_req("Hi, what's up?")) == Intent.INTERACTIVE


def test_code_block_is_code():
    req = _req("Here is some code: ```def foo(): return 1```")
    assert classify(req) == Intent.CODE


def test_error_keywords_are_code():
    assert classify(_req("I got an exception in my loop")) == Intent.CODE


def test_batch_prefix_is_batch():
    assert classify(_req("Summarize the following document into bullet points")) == Intent.BATCH


def test_long_reasoning_prompt_is_reasoning():
    text = (
        "Explain step by step why the CAP theorem implies that a distributed "
        "system cannot simultaneously offer consistency, availability, and "
        "partition tolerance, and analyze how real systems trade off these "
        "properties in practice."
    )
    assert classify(_req(text)) == Intent.REASONING


def test_multiple_reasoning_keywords_short_prompt_is_reasoning():
    assert classify(_req("Explain and analyze this")) == Intent.REASONING
