"""Distributed token-bucket rate limiter.

The ``TokenBucket`` class is the pure algorithmic core. It is intentionally
pure Python with no I/O so it can be unit tested deterministically by
injecting a fake clock.

The enclosing ``RateLimiterStore`` manages many buckets keyed by
(tenant_id, provider) and is safe to call concurrently because every
mutation is serialised through a single ``threading.Lock``. This models the
"leader replica holds authoritative state" pattern: a single replica owns
the counters; follower replicas would replicate them via an RPC replication
log (see ``replication.py``).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# A small type alias so tests can inject a deterministic clock.
Clock = Callable[[], float]


@dataclass
class TokenBucket:
    """Classic token-bucket.

    capacity   -- maximum tokens the bucket can hold (burst size)
    refill_rate -- tokens added per second
    tokens     -- current token level (float; tokens refill continuously)
    updated_at -- the timestamp at which ``tokens`` was last recalculated.
                  ``None`` indicates the bucket has not been observed yet;
                  the first observation seeds it to ``capacity``.
    """

    capacity: float
    refill_rate: float
    tokens: float = field(default=0.0)
    updated_at: Optional[float] = field(default=None)

    def _refill(self, now: float) -> None:
        if self.updated_at is None:
            # First observation: start full. This matches the behaviour of
            # most real gateways that grant new tenants a full burst budget.
            self.tokens = self.capacity
            self.updated_at = now
            return
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.updated_at = now

    def try_consume(self, amount: float, now: float) -> tuple[bool, float, int]:
        """Attempt to take ``amount`` tokens out of the bucket.

        Returns ``(allowed, remaining, retry_after_ms)``.
        """
        self._refill(now)
        if self.tokens >= amount:
            self.tokens -= amount
            return True, self.tokens, 0
        # Not enough tokens: compute how long until we would have enough.
        deficit = amount - self.tokens
        if self.refill_rate <= 0:
            retry_after_ms = 10_000  # effectively never
        else:
            retry_after_ms = int((deficit / self.refill_rate) * 1000) + 1
        return False, self.tokens, retry_after_ms


@dataclass
class BucketConfig:
    capacity: float
    refill_rate: float


class RateLimiterStore:
    """Threadsafe store of named token buckets.

    ``configs`` is a mapping from a key (e.g. ``"tenant_id|provider"``) to
    its bucket configuration. If a key has no configured bucket the store
    falls back to ``default_config``.
    """

    def __init__(
        self,
        default_config: BucketConfig,
        configs: Optional[dict[str, BucketConfig]] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self._default = default_config
        self._configs: dict[str, BucketConfig] = dict(configs or {})
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._clock: Clock = clock or time.monotonic
        # replication log: operations applied in order. Followers replay.
        self._log: list[tuple[float, str, float, bool]] = []
        self._leader_id: str = "leader-0"

    @property
    def leader_id(self) -> str:
        return self._leader_id

    def set_config(self, key: str, config: BucketConfig) -> None:
        with self._lock:
            self._configs[key] = config
            # drop existing bucket so it is recreated under the new config
            self._buckets.pop(key, None)

    def _ensure_bucket(self, key: str) -> TokenBucket:
        bucket = self._buckets.get(key)
        if bucket is None:
            cfg = self._configs.get(key, self._default)
            bucket = TokenBucket(capacity=cfg.capacity, refill_rate=cfg.refill_rate)
            self._buckets[key] = bucket
        return bucket

    def try_consume(self, key: str, amount: float = 1.0) -> tuple[bool, float, int]:
        with self._lock:
            now = self._clock()
            bucket = self._ensure_bucket(key)
            allowed, remaining, retry_after = bucket.try_consume(amount, now)
            # Record the decision in the replication log. In a real system this
            # would be shipped to follower replicas via an RPC.
            self._log.append((now, key, amount, allowed))
            return allowed, remaining, retry_after

    def snapshot(self, key: str) -> Optional[tuple[float, float]]:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                return None
            return bucket.tokens, bucket.capacity

    def set_leader(self, leader_id: str) -> None:
        """Update the leader identity (called after an election)."""
        with self._lock:
            self._leader_id = leader_id

    def replay_log_entry(
        self, ts: float, key: str, amount: float, allowed: bool
    ) -> None:
        """Replay a leader's log entry on a follower replica.

        Applies the state change without re-evaluating the bucket, so the
        follower converges toward the leader's state.
        """
        with self._lock:
            bucket = self._ensure_bucket(key)
            if allowed:
                bucket._refill(ts)
                bucket.tokens = max(0.0, bucket.tokens - amount)
                bucket.updated_at = ts
            self._log.append((ts, key, amount, allowed))

    def replication_log(self) -> list[tuple[float, str, float, bool]]:
        """Return a shallow copy of the replication log for followers."""
        with self._lock:
            return list(self._log)

    def log_length(self) -> int:
        """Return the current length of the replication log."""
        with self._lock:
            return len(self._log)
