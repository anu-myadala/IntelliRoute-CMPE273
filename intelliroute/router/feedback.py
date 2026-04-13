"""Feedback collector for closed-loop routing optimisation.

Accumulates per-provider performance metrics using an exponential moving
average (EMA).  The routing policy consults these live metrics so that
provider scores adapt to observed behaviour rather than relying solely on
static configuration.

The ``FeedbackCollector`` is a pure-logic class with no I/O.  Thread
safety is provided by a single ``threading.Lock``, matching the
conventions used by ``ProviderRegistry``, ``CostAccountant``, and
``RateLimiterStore``.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


Clock = Callable[[], float]


@dataclass
class CompletionOutcome:
    """A single observed completion result fed back from the router."""

    provider: str
    latency_ms: float
    success: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_char_count: int = 1  # avoid division by zero
    response_char_count: int = 0


@dataclass
class ProviderMetrics:
    """Accumulated EMA metrics for a single provider."""

    latency_ema: float = 0.0
    success_rate_ema: float = 1.0
    token_efficiency_ema: float = 1.0
    anomaly_score: float = 0.0
    sample_count: int = 0


class FeedbackCollector:
    """Thread-safe collector of per-provider performance metrics.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor in ``(0, 1]``.  Higher values make the EMA
        more reactive to recent observations.
    clock : callable, optional
        Injectable clock for deterministic testing.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        clock: Optional[Clock] = None,
    ) -> None:
        self._alpha = alpha
        self._clock: Clock = clock or time.monotonic
        self._metrics: dict[str, ProviderMetrics] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, outcome: CompletionOutcome) -> None:
        """Record an observed completion and update EMA metrics."""
        with self._lock:
            m = self._metrics.get(outcome.provider)
            if m is None:
                m = ProviderMetrics()
                self._metrics[outcome.provider] = m

            a = self._alpha

            # Latency EMA (only on success; failures have meaningless latency)
            if outcome.success:
                if m.sample_count == 0:
                    m.latency_ema = outcome.latency_ms
                else:
                    m.latency_ema = (1 - a) * m.latency_ema + a * outcome.latency_ms

            # Success rate EMA
            success_val = 1.0 if outcome.success else 0.0
            if m.sample_count == 0:
                m.success_rate_ema = success_val
            else:
                m.success_rate_ema = (1 - a) * m.success_rate_ema + a * success_val

            # Token efficiency (completion / prompt ratio, success only)
            if outcome.success and outcome.prompt_tokens > 0:
                efficiency = outcome.completion_tokens / outcome.prompt_tokens
                if m.sample_count == 0:
                    m.token_efficiency_ema = efficiency
                else:
                    m.token_efficiency_ema = (
                        (1 - a) * m.token_efficiency_ema + a * efficiency
                    )

            # Anomaly score (hallucination proxy)
            anomaly = self._detect_anomaly(outcome)
            if m.sample_count == 0:
                m.anomaly_score = anomaly
            else:
                m.anomaly_score = (1 - a) * m.anomaly_score + a * anomaly

            m.sample_count += 1

    def get_metrics(self, provider: str) -> Optional[ProviderMetrics]:
        """Return current metrics for *provider*, or ``None`` if unseen."""
        with self._lock:
            m = self._metrics.get(provider)
            if m is None:
                return None
            # Return a copy so callers cannot mutate internal state.
            return ProviderMetrics(
                latency_ema=m.latency_ema,
                success_rate_ema=m.success_rate_ema,
                token_efficiency_ema=m.token_efficiency_ema,
                anomaly_score=m.anomaly_score,
                sample_count=m.sample_count,
            )

    def all_metrics(self) -> dict[str, ProviderMetrics]:
        """Return a snapshot of metrics for every observed provider."""
        with self._lock:
            return {
                name: ProviderMetrics(
                    latency_ema=m.latency_ema,
                    success_rate_ema=m.success_rate_ema,
                    token_efficiency_ema=m.token_efficiency_ema,
                    anomaly_score=m.anomaly_score,
                    sample_count=m.sample_count,
                )
                for name, m in self._metrics.items()
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_anomaly(outcome: CompletionOutcome) -> float:
        """Hallucination proxy based on response-to-prompt length ratio.

        Returns a score in ``[0, 1]``.  A ratio in the *normal* band
        ``[0.1, 10.0]`` scores 0.  Ratios outside this band scale
        linearly toward 1.0.

        Failed completions (no response) always score 0 because there is
        no output to evaluate.
        """
        if not outcome.success:
            return 0.0

        prompt_chars = max(outcome.prompt_char_count, 1)
        ratio = outcome.response_char_count / prompt_chars

        if 0.1 <= ratio <= 10.0:
            return 0.0

        # Too short
        if ratio < 0.1:
            # 0.0 ratio → score 1.0;  0.1 ratio → score 0.0
            return min(1.0, (0.1 - ratio) / 0.1)

        # Too long (ratio > 10.0)
        # 10 → score 0.0;  60 → score 1.0
        return min(1.0, (ratio - 10.0) / 50.0)
