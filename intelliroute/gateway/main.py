"""API Gateway service.

The gateway is the public-facing entry point for IntelliRoute. It:

1. Authenticates clients via a simple X-API-Key header (tenant lookup).
2. Forwards requests to the router, which owns the orchestration logic.
3. Exposes read-only passthroughs for cost summaries and system health
   (so clients/dashboards don't have to speak to internal services).

In a production deployment this would also do:
- Input validation / schema enforcement (handled here via Pydantic)
- Request tracing propagation (we add an X-Request-Id header)
- Load balancing across router replicas (here: single replica)
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException

from ..common.config import settings
from ..common.logging import get_logger, log_event
from ..common.models import CompletionRequest, CompletionResponse, CostSummary

log = get_logger("gateway")

# Very simple API key -> tenant mapping. A real deployment would back this
# with a database and rotating secrets.
API_KEYS: dict[str, str] = {
    os.environ.get("INTELLIROUTE_DEMO_KEY", "demo-key-123"): "demo-tenant",
    os.environ.get("INTELLIROUTE_VIP_KEY", "vip-key-456"): "vip-tenant",
}

app = FastAPI(title="IntelliRoute Gateway")
_http: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def _startup() -> None:
    global _http
    _http = httpx.AsyncClient(timeout=10.0)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _http is not None:
        await _http.aclose()


def _auth(api_key: Optional[str]) -> str:
    if api_key is None or api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    return API_KEYS[api_key]


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "service": "gateway"}


@app.post("/v1/complete", response_model=CompletionResponse)
async def complete(
    req: CompletionRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
) -> CompletionResponse:
    tenant = _auth(x_api_key)
    # Override tenant_id from the authenticated principal, not from the body.
    authoritative = req.model_copy(update={"tenant_id": tenant})

    trace_id = x_request_id or str(uuid.uuid4())
    log_event(log, "incoming_request", trace_id=trace_id, tenant=tenant)

    assert _http is not None
    r = await _http.post(
        f"{settings.router_url}/complete",
        json=authoritative.model_dump(),
        headers={"X-Request-Id": trace_id},
    )
    if r.status_code != 200:
        detail = r.json().get("detail", "router error") if _is_json(r) else "router error"
        raise HTTPException(status_code=r.status_code, detail=detail)
    return CompletionResponse(**r.json())


@app.get("/v1/cost/summary", response_model=CostSummary)
async def cost_summary(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> CostSummary:
    tenant = _auth(x_api_key)
    assert _http is not None
    r = await _http.get(f"{settings.cost_tracker_url}/summary/{tenant}")
    r.raise_for_status()
    return CostSummary(**r.json())


@app.get("/v1/system/health")
async def system_health() -> dict:
    """Aggregate health view: proxies to the health monitor snapshot."""
    assert _http is not None
    try:
        r = await _http.get(f"{settings.health_monitor_url}/snapshot")
        snapshot = r.json() if r.status_code == 200 else {}
    except Exception:
        snapshot = {}
    return {"providers": snapshot}


def _is_json(r: httpx.Response) -> bool:
    ct = r.headers.get("content-type", "")
    return "application/json" in ct
