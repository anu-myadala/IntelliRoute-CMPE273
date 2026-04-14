"""Health monitor service.

Tracks per-provider circuit breakers and publishes a snapshot via HTTP
so the router can consult it when ranking candidates. The router also
reports success/failure back to this service after each attempted call,
which is how the breakers trip.
"""
from __future__ import annotations

import asyncio
import time

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..common.logging import get_logger, log_event
from ..common.models import ProviderHealth
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig

log = get_logger("health_monitor")

breakers: dict[str, CircuitBreaker] = {}
provider_urls: dict[str, str] = {}
_config = CircuitBreakerConfig(failure_threshold=3, open_duration_s=5, half_open_success_required=2)

app = FastAPI(title="IntelliRoute HealthMonitor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_breaker(name: str) -> CircuitBreaker:
    b = breakers.get(name)
    if b is None:
        b = CircuitBreaker(config=_config)
        breakers[name] = b
    return b


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.post("/register")
async def register(name: str, url: str) -> dict:
    provider_urls[name] = url
    _get_breaker(name)
    return {"registered": name}


@app.post("/report/{provider}")
async def report(provider: str, success: bool, latency_ms: float = 0.0) -> dict:
    b = _get_breaker(provider)
    if success:
        b.record_success()
    else:
        b.record_failure()
    log_event(log, "breaker_report", provider=provider, success=success, state=b.state.value)
    return {"provider": provider, "state": b.state.value, "error_rate": b.error_rate()}


@app.get("/snapshot")
async def snapshot() -> dict[str, ProviderHealth]:
    result: dict[str, ProviderHealth] = {}
    now = time.time()
    for name, b in breakers.items():
        result[name] = ProviderHealth(
            name=name,
            healthy=b.state.value != "open",
            error_rate=round(b.error_rate(), 3),
            circuit_state=b.state.value,
            consecutive_failures=b.consecutive_failures,
            last_checked_unix=now,
        )
    return result


@app.get("/snapshot/{provider}", response_model=ProviderHealth)
async def snapshot_one(provider: str) -> ProviderHealth:
    b = _get_breaker(provider)
    return ProviderHealth(
        name=provider,
        healthy=b.state.value != "open",
        error_rate=round(b.error_rate(), 3),
        circuit_state=b.state.value,
        consecutive_failures=b.consecutive_failures,
        last_checked_unix=time.time(),
    )


async def _poll_loop() -> None:
    """Background task: ping registered providers and record liveness."""
    async with httpx.AsyncClient(timeout=2.0) as client:
        while True:
            await asyncio.sleep(5.0)
            for name, url in list(provider_urls.items()):
                b = _get_breaker(name)
                try:
                    r = await client.get(f"{url}/health")
                    if r.status_code == 200 and r.json().get("status") == "healthy":
                        b.record_success()
                    else:
                        b.record_failure()
                except Exception:
                    b.record_failure()


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_poll_loop())
