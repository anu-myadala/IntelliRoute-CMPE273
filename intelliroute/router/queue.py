"""Priority request queue with backpressure and load shedding.

Implements asynchronous request queueing with three priority levels.
High-priority requests (interactive, code) bypass the queue and are
executed immediately.  Medium and low priority requests are enqueued
and processed by background workers.

Load shedding drops the lowest-priority (batch) requests first when
the queue is under pressure, protecting interactive workloads.

The ``RequestQueue`` uses ``asyncio.PriorityQueue`` for ordering and
``threading.Lock`` for the stats counters (matching the project
convention for thread-safe pure-logic classes).
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional

from ..common.models import CompletionRequest, Intent


Clock = Callable[[], float]


class Priority(IntEnum):
    """Request priority levels.  Lower value = higher priority."""

    HIGH = 0    # interactive, code
    MEDIUM = 1  # reasoning
    LOW = 2     # batch


@dataclass
class QueueConfig:
    """Tuning knobs for the request queue."""

    max_depth: int = 100          # total items across all priorities
    max_low_priority: int = 50    # cap on LOW items
    shed_threshold: int = 80      # start shedding LOW when total > this
    timeout_ms: int = 30_000      # max time a request waits in queue


@dataclass(order=True)
class QueuedRequest:
    """An item in the priority queue.

    ``order=True`` uses ``priority`` (then ``enqueued_at``) for
    ``asyncio.PriorityQueue`` ordering: lower numeric priority is
    dequeued first, ties broken by FIFO.
    """

    priority: Priority
    enqueued_at: float
    request_id: str = field(compare=False)
    request: CompletionRequest = field(compare=False)
    future: asyncio.Future = field(compare=False, repr=False)


@dataclass
class QueueStats:
    """Snapshot of queue state for observability."""

    total_depth: int = 0
    by_priority: dict[str, int] = field(default_factory=dict)
    shed_count: int = 0
    timeout_count: int = 0


# Maps Intent -> Priority
_INTENT_PRIORITY: dict[Intent, Priority] = {
    Intent.INTERACTIVE: Priority.HIGH,
    Intent.CODE: Priority.HIGH,
    Intent.REASONING: Priority.MEDIUM,
    Intent.BATCH: Priority.LOW,
}


class RequestQueue:
    """Async-aware priority queue with load shedding.

    Parameters
    ----------
    config : QueueConfig, optional
        Queue tuning parameters.
    clock : callable, optional
        Injectable clock for deterministic testing.
    """

    def __init__(
        self,
        config: Optional[QueueConfig] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self._config = config or QueueConfig()
        self._clock: Clock = clock or time.monotonic
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        # Counters protected by a lock for thread-safe stats.
        self._lock = threading.Lock()
        self._total: int = 0
        self._by_priority: dict[Priority, int] = {p: 0 for p in Priority}
        self._shed_count: int = 0
        self._timeout_count: int = 0

    @staticmethod
    def intent_to_priority(intent: Intent) -> Priority:
        """Map an intent enum to its priority level."""
        return _INTENT_PRIORITY.get(intent, Priority.MEDIUM)

    def try_enqueue(
        self,
        request: CompletionRequest,
        request_id: str,
        intent: Intent,
    ) -> tuple[bool, Optional[QueuedRequest], str]:
        """Attempt to enqueue a request.

        Returns ``(accepted, queued_item, reason)``.

        * HIGH / MEDIUM: accepted unless ``total >= max_depth``.
        * LOW: rejected if ``total >= shed_threshold`` OR
          ``low_count >= max_low_priority``.
        """
        priority = self.intent_to_priority(intent)
        cfg = self._config

        with self._lock:
            # Absolute cap
            if self._total >= cfg.max_depth:
                return False, None, "queue_full"

            # Load-shedding: shed LOW priority first
            if priority == Priority.LOW:
                if self._total >= cfg.shed_threshold:
                    self._shed_count += 1
                    return False, None, "load_shed"
                if self._by_priority[Priority.LOW] >= cfg.max_low_priority:
                    self._shed_count += 1
                    return False, None, "load_shed"

            loop = asyncio.get_event_loop()
            future = loop.create_future()
            item = QueuedRequest(
                priority=priority,
                enqueued_at=self._clock(),
                request_id=request_id,
                request=request,
                future=future,
            )
            self._queue.put_nowait(item)
            self._total += 1
            self._by_priority[priority] += 1

        return True, item, ""

    async def dequeue(self) -> QueuedRequest:
        """Block until a request is available.  Highest priority first."""
        item: QueuedRequest = await self._queue.get()
        with self._lock:
            self._total -= 1
            self._by_priority[item.priority] -= 1
        return item

    def record_timeout(self) -> None:
        """Increment the timeout counter (called by the worker on timeout)."""
        with self._lock:
            self._timeout_count += 1

    def stats(self) -> QueueStats:
        """Return a snapshot of queue statistics."""
        with self._lock:
            return QueueStats(
                total_depth=self._total,
                by_priority={p.name: self._by_priority[p] for p in Priority},
                shed_count=self._shed_count,
                timeout_count=self._timeout_count,
            )
