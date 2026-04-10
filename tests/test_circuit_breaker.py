"""Unit tests for the circuit breaker."""
from __future__ import annotations

from intelliroute.health_monitor.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)


class _Clock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _make_breaker(**kwargs) -> tuple[CircuitBreaker, _Clock]:
    clock = _Clock()
    breaker = CircuitBreaker(
        config=CircuitBreakerConfig(**kwargs),
        _clock=clock,  # type: ignore[arg-type]
    )
    return breaker, clock


def test_breaker_starts_closed_and_allows_requests():
    breaker, _ = _make_breaker()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_breaker_opens_after_threshold_failures():
    breaker, clock = _make_breaker(failure_threshold=3, open_duration_s=5)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False


def test_breaker_transitions_to_half_open_after_cooldown():
    breaker, clock = _make_breaker(failure_threshold=2, open_duration_s=5)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    # Still within cooldown
    clock.now = 1.0
    assert breaker.allow_request() is False

    # Cooldown elapsed -> half open
    clock.now = 10.0
    assert breaker.allow_request() is True
    assert breaker.state == CircuitState.HALF_OPEN


def test_breaker_closes_after_successful_trials():
    breaker, clock = _make_breaker(
        failure_threshold=2, open_duration_s=5, half_open_success_required=2
    )
    breaker.record_failure()
    breaker.record_failure()
    clock.now = 10.0
    breaker.allow_request()  # transitions to HALF_OPEN
    breaker.record_success()
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED


def test_breaker_reopens_on_failure_during_half_open():
    breaker, clock = _make_breaker(
        failure_threshold=2, open_duration_s=5, half_open_success_required=3
    )
    breaker.record_failure()
    breaker.record_failure()
    clock.now = 10.0
    breaker.allow_request()  # HALF_OPEN
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN


def test_error_rate_reflects_sliding_window():
    breaker, _ = _make_breaker(failure_threshold=1000, window_size=4)
    breaker.record_success()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.error_rate() == 0.5
    breaker.record_failure()
    # Window dropped the oldest success; now 3 failures / 4.
    assert breaker.error_rate() == 0.75
