"""Unit tests for the multi-objective routing policy."""
from __future__ import annotations

from intelliroute.common.models import Intent, ProviderHealth, ProviderInfo
from intelliroute.router.feedback import CompletionOutcome, FeedbackCollector
from intelliroute.router.policy import RoutingPolicy


def _providers() -> list[ProviderInfo]:
    return [
        ProviderInfo(
            name="fast",
            url="http://fast",
            model="fast-1",
            capability={"interactive": 0.8, "reasoning": 0.4, "batch": 0.5, "code": 0.6},
            cost_per_1k_tokens=0.002,
            typical_latency_ms=150,
        ),
        ProviderInfo(
            name="smart",
            url="http://smart",
            model="smart-1",
            capability={"interactive": 0.7, "reasoning": 0.95, "batch": 0.8, "code": 0.9},
            cost_per_1k_tokens=0.02,
            typical_latency_ms=900,
        ),
        ProviderInfo(
            name="cheap",
            url="http://cheap",
            model="cheap-1",
            capability={"interactive": 0.5, "reasoning": 0.4, "batch": 0.7, "code": 0.4},
            cost_per_1k_tokens=0.0005,
            typical_latency_ms=700,
        ),
    ]


def test_interactive_prefers_low_latency():
    policy = RoutingPolicy()
    ranked = policy.rank(_providers(), health={}, intent=Intent.INTERACTIVE)
    assert ranked[0].provider.name == "fast"


def test_reasoning_prefers_capability():
    policy = RoutingPolicy()
    ranked = policy.rank(_providers(), health={}, intent=Intent.REASONING)
    assert ranked[0].provider.name == "smart"


def test_batch_prefers_cheapest():
    policy = RoutingPolicy()
    ranked = policy.rank(_providers(), health={}, intent=Intent.BATCH)
    assert ranked[0].provider.name == "cheap"


def test_open_circuit_demoted_to_last():
    policy = RoutingPolicy()
    health = {
        "fast": ProviderHealth(name="fast", healthy=False, circuit_state="open"),
    }
    ranked = policy.rank(_providers(), health=health, intent=Intent.INTERACTIVE)
    names = [s.provider.name for s in ranked]
    # 'fast' must not be first when its breaker is open; smart/cheap come first.
    assert names[0] != "fast"
    # 'fast' should not be in the usable list at all (only surfaces as fallback
    # when every provider is open; here others are healthy).
    assert "fast" not in names


def test_latency_budget_zeros_out_slow_providers():
    policy = RoutingPolicy()
    ranked = policy.rank(
        _providers(),
        health={},
        intent=Intent.INTERACTIVE,
        latency_budget_ms=200,
    )
    # Only "fast" meets the 200 ms budget, so it must be first.
    assert ranked[0].provider.name == "fast"


def test_ranking_is_deterministic_and_stable():
    policy = RoutingPolicy()
    r1 = policy.rank(_providers(), health={}, intent=Intent.CODE)
    r2 = policy.rank(_providers(), health={}, intent=Intent.CODE)
    assert [s.provider.name for s in r1] == [s.provider.name for s in r2]


def test_empty_provider_list_returns_empty():
    policy = RoutingPolicy()
    assert policy.rank([], health={}, intent=Intent.INTERACTIVE) == []


# ---- Feedback integration tests ----


def test_policy_uses_feedback_latency_over_static():
    """When feedback reports different latencies than static config, the
    ranking should shift accordingly."""
    fc = FeedbackCollector(alpha=1.0)
    # Record very fast latency for 'smart' (static is 900ms)
    fc.record(CompletionOutcome(
        provider="smart", latency_ms=50.0, success=True,
        prompt_tokens=50, completion_tokens=25,
        prompt_char_count=200, response_char_count=100,
    ))
    # Record degraded latency for 'fast' (static is 150ms)
    fc.record(CompletionOutcome(
        provider="fast", latency_ms=800.0, success=True,
        prompt_tokens=50, completion_tokens=25,
        prompt_char_count=200, response_char_count=100,
    ))

    policy_fb = RoutingPolicy(feedback=fc)
    policy_no = RoutingPolicy()

    ranked_fb = policy_fb.rank(_providers(), health={}, intent=Intent.INTERACTIVE)
    ranked_no = policy_no.rank(_providers(), health={}, intent=Intent.INTERACTIVE)

    # Without feedback, fast wins interactive (lowest static latency)
    assert ranked_no[0].provider.name == "fast"
    # With feedback, fast's latency score plummets and smart overtakes it
    assert ranked_fb[0].provider.name != "fast"


def test_policy_penalises_anomalous_provider():
    """A provider with a high anomaly score should be demoted."""
    fc = FeedbackCollector(alpha=1.0)
    # Record an extremely anomalous response for 'fast'
    fc.record(CompletionOutcome(
        provider="fast", latency_ms=50.0, success=True,
        prompt_tokens=50, completion_tokens=25,
        prompt_char_count=10, response_char_count=10000,  # very long vs prompt
    ))
    policy_fb = RoutingPolicy(feedback=fc)
    policy_no = RoutingPolicy()  # no feedback

    ranked_fb = policy_fb.rank(_providers(), health={}, intent=Intent.INTERACTIVE)
    ranked_no = policy_no.rank(_providers(), health={}, intent=Intent.INTERACTIVE)

    # With no feedback, fast is first for interactive
    assert ranked_no[0].provider.name == "fast"
    # Score for fast should be lower with the anomaly penalty
    fast_score_fb = next(s.score for s in ranked_fb if s.provider.name == "fast")
    fast_score_no = next(s.score for s in ranked_no if s.provider.name == "fast")
    assert fast_score_fb < fast_score_no


def test_policy_falls_back_to_static_without_feedback():
    """When feedback is None, the policy behaves identically to before."""
    policy = RoutingPolicy(feedback=None)
    ranked = policy.rank(_providers(), health={}, intent=Intent.INTERACTIVE)
    assert ranked[0].provider.name == "fast"
