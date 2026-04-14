"""Cost tracker HTTP service.

This is the asynchronous-accounting side of IntelliRoute. The router
posts cost events to ``/events`` (fire-and-forget from the request hot
path), and the tracker maintains running per-tenant aggregates that the
gateway can query via ``/summary/{tenant}``.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..common.logging import get_logger, log_event
from ..common.models import CostEvent, CostSummary
from .accounting import CostAccountant

log = get_logger("cost_tracker")
accountant = CostAccountant()
app = FastAPI(title="IntelliRoute CostTracker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.post("/events")
async def record(event: CostEvent) -> dict:
    accountant.record(event)
    log_event(
        log,
        "cost_event_recorded",
        tenant=event.tenant_id,
        provider=event.provider,
        cost=event.estimated_cost_usd,
    )
    return {"ok": True}


@app.get("/summary/{tenant_id}", response_model=CostSummary)
async def summary(tenant_id: str) -> CostSummary:
    return accountant.summary(tenant_id)


class Budget(BaseModel):
    tenant_id: str
    budget_usd: float


@app.post("/budget")
async def set_budget(b: Budget) -> dict:
    accountant.set_budget(b.tenant_id, b.budget_usd)
    return {"tenant": b.tenant_id, "budget_usd": b.budget_usd}


@app.get("/alerts")
async def alerts() -> dict:
    return {"alerts": accountant.alerts()}
