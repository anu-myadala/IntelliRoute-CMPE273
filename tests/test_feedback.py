"""Unit tests for the feedback collector."""
from intelliroute.router.feedback import CompletionOutcome, FeedbackCollector


def _outcome(
    provider: str = "p1",
    latency_ms: float = 100.0,
    success: bool = True,
    prompt_tokens: int = 50,
    completion_tokens: int = 25,
    prompt_chars: int = 200,
    response_chars: int = 100,
) -> CompletionOutcome:
    return CompletionOutcome(
        provider=provider,
        latency_ms=latency_ms,
        success=success,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_char_count=prompt_chars,
        response_char_count=response_chars,
    )


def test_first_record_initialises_ema():
    fc = FeedbackCollector(alpha=0.2)
    fc.record(_outcome(latency_ms=200.0))
    m = fc.get_metrics("p1")
    assert m is not None
    assert m.latency_ema == 200.0
    assert m.success_rate_ema == 1.0
    assert m.sample_count == 1


def test_ema_converges_toward_recent_values():
    fc = FeedbackCollector(alpha=0.5)
    fc.record(_outcome(latency_ms=100.0))
    fc.record(_outcome(latency_ms=200.0))
    m = fc.get_metrics("p1")
    assert m is not None
    # EMA: 100 -> 0.5*100 + 0.5*200 = 150
    assert abs(m.latency_ema - 150.0) < 0.01


def test_success_rate_drops_on_failures():
    fc = FeedbackCollector(alpha=0.5)
    fc.record(_outcome(success=True))
    fc.record(_outcome(success=False))
    m = fc.get_metrics("p1")
    assert m is not None
    # EMA: 1.0 -> 0.5*1.0 + 0.5*0.0 = 0.5
    assert abs(m.success_rate_ema - 0.5) < 0.01


def test_anomaly_flags_very_short_response():
    fc = FeedbackCollector(alpha=1.0)
    fc.record(_outcome(prompt_chars=1000, response_chars=5))
    m = fc.get_metrics("p1")
    assert m is not None
    assert m.anomaly_score > 0.5


def test_anomaly_flags_very_long_response():
    fc = FeedbackCollector(alpha=1.0)
    fc.record(_outcome(prompt_chars=10, response_chars=1000))
    m = fc.get_metrics("p1")
    assert m is not None
    assert m.anomaly_score > 0.5


def test_normal_response_scores_zero_anomaly():
    fc = FeedbackCollector(alpha=1.0)
    fc.record(_outcome(prompt_chars=200, response_chars=200))
    m = fc.get_metrics("p1")
    assert m is not None
    assert m.anomaly_score == 0.0


def test_unseen_provider_returns_none():
    fc = FeedbackCollector()
    assert fc.get_metrics("unknown") is None


def test_all_metrics_snapshot():
    fc = FeedbackCollector()
    fc.record(_outcome(provider="a"))
    fc.record(_outcome(provider="b"))
    snapshot = fc.all_metrics()
    assert set(snapshot.keys()) == {"a", "b"}
    assert snapshot["a"].sample_count == 1
    assert snapshot["b"].sample_count == 1


def test_multiple_providers_isolated():
    fc = FeedbackCollector(alpha=1.0)
    fc.record(_outcome(provider="fast", latency_ms=50.0))
    fc.record(_outcome(provider="slow", latency_ms=500.0))
    assert fc.get_metrics("fast").latency_ema == 50.0
    assert fc.get_metrics("slow").latency_ema == 500.0


def test_token_efficiency_computed():
    fc = FeedbackCollector(alpha=1.0)
    fc.record(_outcome(prompt_tokens=100, completion_tokens=50))
    m = fc.get_metrics("p1")
    assert m is not None
    assert abs(m.token_efficiency_ema - 0.5) < 0.01
