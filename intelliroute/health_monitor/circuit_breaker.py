"""Per-provider circuit breaker.

Implements the standard three-state breaker:

    closed  -> normal operation, counting failures in a sliding window
    open    -> all requests fail fast for ``open_duration_s`` seconds
    half_open -> one trial request allowed; success closes, failure opens

The implementation is intentionally pure so it can be unit tested with an
injected clock; the health monitor service wraps a dict of these keyed by
provider name.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


Clock = Callable[[], float]


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # failures before opening
    open_duration_s: float = 10.0       # how long to stay open before trialing
    half_open_success_required: int = 2  # successful trials needed to close
    window_size: int = 20               # sliding window of recent outcomes


@dataclass
class CircuitBreaker:
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    half_open_successes: int = 0
    opened_at: float = 0.0
    _window: list[bool] = field(default_factory=list)  # True = success
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _clock: Clock = field(default_factory=lambda: __import__("time").monotonic)

    def allow_request(self, now: Optional[float] = None) -> bool:
        with self._lock:
            now = now if now is not None else self._clock()
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                if now - self.opened_at >= self.config.open_duration_s:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_successes = 0
                    return True
                return False
            # HALF_OPEN: allow probe requests through.
            return True

    def record_success(self, now: Optional[float] = None) -> None:
        with self._lock:
            now = now if now is not None else self._clock()
            self._window.append(True)
            if len(self._window) > self.config.window_size:
                self._window.pop(0)
            self.consecutive_failures = 0
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_successes += 1
                if self.half_open_successes >= self.config.half_open_success_required:
                    self.state = CircuitState.CLOSED
                    self.half_open_successes = 0
            # CLOSED stays CLOSED; OPEN cannot reach here because requests are
            # supposed to be rejected via allow_request().

    def record_failure(self, now: Optional[float] = None) -> None:
        with self._lock:
            now = now if now is not None else self._clock()
            self._window.append(False)
            if len(self._window) > self.config.window_size:
                self._window.pop(0)
            self.consecutive_failures += 1

            if self.state == CircuitState.HALF_OPEN:
                # trial failed: reopen immediately
                self.state = CircuitState.OPEN
                self.opened_at = now
                self.half_open_successes = 0
                return

            if (
                self.state == CircuitState.CLOSED
                and self.consecutive_failures >= self.config.failure_threshold
            ):
                self.state = CircuitState.OPEN
                self.opened_at = now

    def error_rate(self) -> float:
        with self._lock:
            if not self._window:
                return 0.0
            failures = sum(1 for ok in self._window if not ok)
            return failures / len(self._window)
