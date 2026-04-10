"""In-memory cost accounting.

This is the eventually-consistent side of the system. Cost events are
published asynchronously from the router after a successful completion;
the cost tracker consumes them and maintains a running aggregate per
tenant.

In a production system this would be backed by a durable append-only log
(Kafka, NATS JetStream) and a columnar store for the rollups. For the
course project we keep it in-memory: the aggregation logic is still
representative and fully tested.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field

from ..common.models import CostEvent, CostSummary


@dataclass
class _TenantRollup:
    total_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    by_provider: dict[str, float] = field(default_factory=lambda: defaultdict(float))


class CostAccountant:
    def __init__(self, budgets: dict[str, float] | None = None) -> None:
        self._rollups: dict[str, _TenantRollup] = defaultdict(_TenantRollup)
        self._budgets: dict[str, float] = dict(budgets or {})
        self._alerts: list[str] = []
        # Set of tenants that have already crossed their budget; prevents
        # alert spam once the threshold has been breached.
        self._crossed: set[str] = set()
        self._lock = threading.Lock()

    def set_budget(self, tenant_id: str, budget_usd: float) -> None:
        with self._lock:
            self._budgets[tenant_id] = budget_usd

    def record(self, event: CostEvent) -> None:
        with self._lock:
            r = self._rollups[event.tenant_id]
            r.total_requests += 1
            r.total_tokens += event.prompt_tokens + event.completion_tokens
            r.total_cost_usd += event.estimated_cost_usd
            r.by_provider[event.provider] += event.estimated_cost_usd

            # Budget alerting: fire exactly once per tenant per crossing.
            budget = self._budgets.get(event.tenant_id)
            if (
                budget is not None
                and r.total_cost_usd > budget
                and event.tenant_id not in self._crossed
            ):
                self._crossed.add(event.tenant_id)
                self._alerts.append(
                    f"BUDGET_EXCEEDED tenant={event.tenant_id} "
                    f"spent={r.total_cost_usd:.4f} budget={budget:.4f}"
                )

    def summary(self, tenant_id: str) -> CostSummary:
        with self._lock:
            r = self._rollups.get(tenant_id)
            if r is None:
                return CostSummary(
                    tenant_id=tenant_id,
                    total_requests=0,
                    total_tokens=0,
                    total_cost_usd=0.0,
                    by_provider={},
                )
            return CostSummary(
                tenant_id=tenant_id,
                total_requests=r.total_requests,
                total_tokens=r.total_tokens,
                total_cost_usd=round(r.total_cost_usd, 6),
                by_provider={k: round(v, 6) for k, v in r.by_provider.items()},
            )

    def alerts(self) -> list[str]:
        with self._lock:
            return list(self._alerts)
