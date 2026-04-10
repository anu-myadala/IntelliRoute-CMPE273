"""Unit tests for the cost accountant."""
from __future__ import annotations

from intelliroute.common.models import CostEvent
from intelliroute.cost_tracker.accounting import CostAccountant


def _event(tenant: str, provider: str, cost: float, tokens: int = 100, ts: float = 0.0) -> CostEvent:
    return CostEvent(
        request_id=f"req-{ts}-{tenant}",
        tenant_id=tenant,
        provider=provider,
        model="m1",
        prompt_tokens=tokens // 2,
        completion_tokens=tokens // 2,
        estimated_cost_usd=cost,
        unix_ts=ts,
    )


def test_records_and_rolls_up_by_provider():
    acc = CostAccountant()
    acc.record(_event("t1", "fast", cost=0.01, tokens=100))
    acc.record(_event("t1", "smart", cost=0.05, tokens=200))
    acc.record(_event("t1", "fast", cost=0.02, tokens=100))

    s = acc.summary("t1")
    assert s.total_requests == 3
    assert s.total_tokens == 400
    assert round(s.total_cost_usd, 4) == 0.08
    assert round(s.by_provider["fast"], 4) == 0.03
    assert round(s.by_provider["smart"], 4) == 0.05


def test_unknown_tenant_returns_zero_summary():
    acc = CostAccountant()
    s = acc.summary("ghost")
    assert s.total_requests == 0
    assert s.total_cost_usd == 0.0
    assert s.by_provider == {}


def test_budget_alert_fires_once_on_crossing():
    acc = CostAccountant(budgets={"t1": 0.05})
    acc.record(_event("t1", "fast", cost=0.03))
    assert acc.alerts() == []
    acc.record(_event("t1", "smart", cost=0.03))  # total 0.06 > 0.05
    assert len(acc.alerts()) == 1
    acc.record(_event("t1", "smart", cost=0.01))
    # Alert string is deduplicated so we still only have one unique alert.
    assert len(acc.alerts()) == 1


def test_tenants_are_isolated():
    acc = CostAccountant()
    acc.record(_event("t1", "fast", cost=0.01))
    acc.record(_event("t2", "fast", cost=0.05))
    assert acc.summary("t1").total_cost_usd == 0.01
    assert acc.summary("t2").total_cost_usd == 0.05
