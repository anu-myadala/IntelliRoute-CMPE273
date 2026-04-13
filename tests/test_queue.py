"""Unit tests for the priority request queue."""
from __future__ import annotations

import asyncio

import pytest

from intelliroute.common.models import ChatMessage, CompletionRequest, Intent
from intelliroute.router.queue import Priority, QueueConfig, RequestQueue


def _req(tenant: str = "t1", content: str = "hello") -> CompletionRequest:
    return CompletionRequest(
        tenant_id=tenant,
        messages=[ChatMessage(role="user", content=content)],
    )


# ---- Priority mapping ----

def test_interactive_maps_to_high():
    assert RequestQueue.intent_to_priority(Intent.INTERACTIVE) == Priority.HIGH


def test_code_maps_to_high():
    assert RequestQueue.intent_to_priority(Intent.CODE) == Priority.HIGH


def test_reasoning_maps_to_medium():
    assert RequestQueue.intent_to_priority(Intent.REASONING) == Priority.MEDIUM


def test_batch_maps_to_low():
    assert RequestQueue.intent_to_priority(Intent.BATCH) == Priority.LOW


# ---- Enqueue / shedding ----

@pytest.mark.asyncio
async def test_enqueue_accepts_within_capacity():
    q = RequestQueue(config=QueueConfig(max_depth=10, shed_threshold=8))
    ok, item, reason = q.try_enqueue(_req(), "r1", Intent.BATCH)
    assert ok is True
    assert item is not None
    assert reason == ""


@pytest.mark.asyncio
async def test_enqueue_rejects_low_at_shed_threshold():
    cfg = QueueConfig(max_depth=10, shed_threshold=2, max_low_priority=10)
    q = RequestQueue(config=cfg)
    # Fill to shed threshold
    q.try_enqueue(_req(), "r1", Intent.REASONING)
    q.try_enqueue(_req(), "r2", Intent.REASONING)
    # Now a LOW priority should be shed
    ok, item, reason = q.try_enqueue(_req(), "r3", Intent.BATCH)
    assert ok is False
    assert reason == "load_shed"


@pytest.mark.asyncio
async def test_enqueue_rejects_all_at_max_depth():
    cfg = QueueConfig(max_depth=2, shed_threshold=10)
    q = RequestQueue(config=cfg)
    q.try_enqueue(_req(), "r1", Intent.INTERACTIVE)
    q.try_enqueue(_req(), "r2", Intent.INTERACTIVE)
    ok, _, reason = q.try_enqueue(_req(), "r3", Intent.INTERACTIVE)
    assert ok is False
    assert reason == "queue_full"


@pytest.mark.asyncio
async def test_medium_not_shed_before_max_depth():
    """MEDIUM priority is never load-shed, only rejected at max_depth."""
    cfg = QueueConfig(max_depth=10, shed_threshold=2)
    q = RequestQueue(config=cfg)
    # Fill past shed_threshold with LOW
    q.try_enqueue(_req(), "r1", Intent.REASONING)
    q.try_enqueue(_req(), "r2", Intent.REASONING)
    q.try_enqueue(_req(), "r3", Intent.REASONING)
    # MEDIUM should still be accepted (shed only targets LOW)
    ok, _, reason = q.try_enqueue(_req(), "r4", Intent.REASONING)
    assert ok is True


@pytest.mark.asyncio
async def test_low_priority_cap():
    cfg = QueueConfig(max_depth=100, shed_threshold=100, max_low_priority=2)
    q = RequestQueue(config=cfg)
    q.try_enqueue(_req(), "r1", Intent.BATCH)
    q.try_enqueue(_req(), "r2", Intent.BATCH)
    ok, _, reason = q.try_enqueue(_req(), "r3", Intent.BATCH)
    assert ok is False
    assert reason == "load_shed"


# ---- Dequeue ordering ----

@pytest.mark.asyncio
async def test_dequeue_returns_highest_priority_first():
    q = RequestQueue(config=QueueConfig(max_depth=10, shed_threshold=10))
    q.try_enqueue(_req(), "low", Intent.BATCH)
    q.try_enqueue(_req(), "high", Intent.INTERACTIVE)
    q.try_enqueue(_req(), "med", Intent.REASONING)

    first = await q.dequeue()
    second = await q.dequeue()
    third = await q.dequeue()

    assert first.request_id == "high"
    assert second.request_id == "med"
    assert third.request_id == "low"


# ---- Stats ----

@pytest.mark.asyncio
async def test_stats_reflect_queue_state():
    cfg = QueueConfig(max_depth=10, shed_threshold=8, max_low_priority=5)
    q = RequestQueue(config=cfg)
    q.try_enqueue(_req(), "r1", Intent.INTERACTIVE)
    q.try_enqueue(_req(), "r2", Intent.BATCH)

    s = q.stats()
    assert s.total_depth == 2
    assert s.by_priority["HIGH"] == 1
    assert s.by_priority["LOW"] == 1


@pytest.mark.asyncio
async def test_shed_count_increments():
    cfg = QueueConfig(max_depth=10, shed_threshold=0, max_low_priority=0)
    q = RequestQueue(config=cfg)
    q.try_enqueue(_req(), "r1", Intent.BATCH)  # shed immediately
    q.try_enqueue(_req(), "r2", Intent.BATCH)

    s = q.stats()
    assert s.shed_count == 2
