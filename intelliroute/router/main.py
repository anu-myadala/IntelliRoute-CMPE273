"""Router service.

Responsibilities
----------------
1. Classify incoming requests into an :class:`Intent`.
2. Consult the provider registry (service discovery) and the health
   monitor (circuit breaker state) to build a ranked list of candidates.
3. Check the rate limiter before attempting a provider.
4. Call the chosen provider over HTTP; on failure, fall back to the next
   provider in the ranked list ("adaptive degradation").
5. Report success/failure to the health monitor and publish a cost
   event to the cost tracker (fire-and-forget).
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..common.config import settings
from ..common.logging import get_logger, log_event
from ..common.models import (
    CompletionRequest,
    CompletionResponse,
    CostEvent,
    ProviderHealth,
    ProviderInfo,
    RateLimitCheck,
)
from .intent import classify
from .policy import RoutingPolicy
from .registry import ProviderRegistry

log = get_logger("router")

registry = ProviderRegistry()
policy = RoutingPolicy()

app = FastAPI(title="IntelliRoute Router")

_http: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def _startup() -> None:
    global _http
    _http = httpx.AsyncClient(timeout=5.0)
    # Auto-register the three mock providers if the env vars are set.
    _bootstrap_mock_registry()


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _http is not None:
        await _http.aclose()


def _bootstrap_mock_registry() -> None:
    """Register the three canonical mock providers from env config."""
    bootstrap = [
        ProviderInfo(
            name="mock-fast",
            url=f"http://{settings.host}:{settings.mock_fast_port}",
            model="fast-1",
            capability={"interactive": 0.85, "reasoning": 0.45, "batch": 0.5, "code": 0.6},
            cost_per_1k_tokens=0.002,
            typical_latency_ms=120,
        ),
        ProviderInfo(
            name="mock-smart",
            url=f"http://{settings.host}:{settings.mock_smart_port}",
            model="smart-1",
            capability={"interactive": 0.7, "reasoning": 0.95, "batch": 0.8, "code": 0.9},
            cost_per_1k_tokens=0.02,
            typical_latency_ms=900,
        ),
        ProviderInfo(
            name="mock-cheap",
            url=f"http://{settings.host}:{settings.mock_cheap_port}",
            model="cheap-1",
            capability={"interactive": 0.55, "reasoning": 0.4, "batch": 0.75, "code": 0.45},
            cost_per_1k_tokens=0.0003,
            typical_latency_ms=600,
        ),
    ]
    if os.environ.get("INTELLIROUTE_SKIP_BOOTSTRAP") != "1":
        registry.bulk_register(bootstrap)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "providers": len(registry.all())}


@app.post("/providers")
async def register_provider(p: ProviderInfo) -> dict:
    registry.register(p)
    return {"registered": p.name}


@app.delete("/providers/{name}")
async def deregister_provider(name: str) -> dict:
    registry.deregister(name)
    return {"deregistered": name}


@app.get("/providers")
async def list_providers() -> list[ProviderInfo]:
    return registry.all()


async def _fetch_health_snapshot() -> dict[str, ProviderHealth]:
    assert _http is not None
    try:
        r = await _http.get(f"{settings.health_monitor_url}/snapshot")
        if r.status_code != 200:
            return {}
        data = r.json()
        return {name: ProviderHealth(**h) for name, h in data.items()}
    except Exception as exc:
        log_event(log, "health_snapshot_failed", error=str(exc))
        return {}


async def _check_rate_limit(tenant: str, provider: str) -> tuple[bool, int]:
    assert _http is not None
    try:
        r = await _http.post(
            f"{settings.rate_limiter_url}/check",
            json=RateLimitCheck(tenant_id=tenant, provider=provider).model_dump(),
        )
        if r.status_code != 200:
            return True, 0  # fail-open if the limiter is down
        data = r.json()
        return bool(data.get("allowed", True)), int(data.get("retry_after_ms", 0))
    except Exception:
        return True, 0


async def _report_health(provider: str, success: bool, latency_ms: float) -> None:
    assert _http is not None
    try:
        await _http.post(
            f"{settings.health_monitor_url}/report/{provider}",
            params={"success": str(success).lower(), "latency_ms": latency_ms},
        )
    except Exception:
        pass


async def _publish_cost(event: CostEvent) -> None:
    assert _http is not None
    try:
        await _http.post(
            f"{settings.cost_tracker_url}/events", json=event.model_dump()
        )
    except Exception:
        pass


async def _call_provider(
    info: ProviderInfo, req: CompletionRequest
) -> tuple[bool, float, dict | None]:
    assert _http is not None
    start = time.monotonic()
    try:
        payload = {
            "messages": [m.model_dump() for m in req.messages],
            "max_tokens": req.max_tokens,
        }
        r = await _http.post(f"{info.url}/v1/chat", json=payload, timeout=5.0)
        elapsed_ms = (time.monotonic() - start) * 1000
        if r.status_code != 200:
            return False, elapsed_ms, None
        return True, elapsed_ms, r.json()
    except Exception:
        return False, (time.monotonic() - start) * 1000, None


class RouteDecision(BaseModel):
    intent: str
    ranked: list[str]
    scores: dict[str, float]


@app.post("/decide", response_model=RouteDecision)
async def decide(req: CompletionRequest) -> RouteDecision:
    """Introspection endpoint: return the routing decision without executing it."""
    intent = classify(req)
    health = await _fetch_health_snapshot()
    ranked = policy.rank(
        registry.all(), health=health, intent=intent, latency_budget_ms=req.latency_budget_ms
    )
    return RouteDecision(
        intent=intent.value,
        ranked=[s.provider.name for s in ranked],
        scores={s.provider.name: round(s.score, 4) for s in ranked},
    )


@app.post("/complete", response_model=CompletionResponse)
async def complete(req: CompletionRequest) -> CompletionResponse:
    request_id = str(uuid.uuid4())
    intent = classify(req)
    health = await _fetch_health_snapshot()
    ranked = policy.rank(
        registry.all(), health=health, intent=intent, latency_budget_ms=req.latency_budget_ms
    )
    if not ranked:
        raise HTTPException(status_code=503, detail="no providers registered")

    log_event(log, "route_decided", request_id=request_id, intent=intent.value,
              primary=ranked[0].provider.name)

    fallback_used = False
    last_error: Optional[str] = None

    for i, scored in enumerate(ranked):
        info = scored.provider
        allowed, retry_ms = await _check_rate_limit(req.tenant_id, info.name)
        if not allowed:
            log_event(log, "rate_limited", provider=info.name, retry_after_ms=retry_ms)
            last_error = f"rate_limited:{info.name}"
            fallback_used = True
            continue

        ok, latency_ms, data = await _call_provider(info, req)
        asyncio.create_task(_report_health(info.name, ok, latency_ms))
        if not ok:
            log_event(log, "provider_failed", provider=info.name, latency_ms=latency_ms)
            last_error = f"provider_failed:{info.name}"
            fallback_used = True
            continue

        prompt_tokens = int(data.get("prompt_tokens", 0))
        completion_tokens = int(data.get("completion_tokens", 0))
        total_tokens = prompt_tokens + completion_tokens
        estimated_cost = (total_tokens / 1000.0) * info.cost_per_1k_tokens

        asyncio.create_task(
            _publish_cost(
                CostEvent(
                    request_id=request_id,
                    tenant_id=req.tenant_id,
                    provider=info.name,
                    model=info.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated_cost_usd=estimated_cost,
                    unix_ts=time.time(),
                )
            )
        )

        return CompletionResponse(
            request_id=request_id,
            provider=info.name,
            model=info.model,
            content=data.get("content", ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            estimated_cost_usd=round(estimated_cost, 6),
            fallback_used=fallback_used or i > 0,
            degraded=i > 0,
        )

    raise HTTPException(status_code=503, detail=f"all providers failed: {last_error}")
