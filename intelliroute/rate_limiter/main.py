"""Rate limiter HTTP service.

Wraps the ``RateLimiterStore`` in a FastAPI app. Gateway/router replicas
call ``/check`` on every outbound request. Per-key configuration can be
updated at runtime via ``/config``.

Leader/follower semantics
-------------------------
The store exposes a replication log. In this simple implementation there
is a single leader replica (the process that owns authoritative counter
state); the ``/leader`` endpoint reports the leader id, and ``/log``
returns the replication log so that follower replicas can tail and
replay operations. Electing a new leader is left as a configuration
change in the course demo.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ..common.logging import get_logger, log_event
from ..common.models import RateLimitCheck, RateLimitResult
from .token_bucket import BucketConfig, RateLimiterStore

log = get_logger("rate_limiter")

# Reasonable defaults: 60 requests/min per (tenant, provider) pair with a
# burst of 10. Tweakable at runtime via /config.
_default = BucketConfig(capacity=10, refill_rate=1.0)
store = RateLimiterStore(default_config=_default)

app = FastAPI(title="IntelliRoute RateLimiter")


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "leader": store.leader_id}


@app.get("/leader")
async def leader() -> dict:
    return {"leader": store.leader_id}


@app.post("/check", response_model=RateLimitResult)
async def check(req: RateLimitCheck) -> RateLimitResult:
    key = f"{req.tenant_id}|{req.provider}"
    allowed, remaining, retry_after = store.try_consume(key, amount=req.tokens_requested)
    log_event(
        log,
        "rate_limit_check",
        key=key,
        allowed=allowed,
        remaining=round(remaining, 3),
        retry_after_ms=retry_after,
    )
    return RateLimitResult(
        allowed=allowed,
        remaining=remaining,
        retry_after_ms=retry_after,
        leader_replica=store.leader_id,
    )


class ConfigPayload(BaseModel):
    key: str
    capacity: float
    refill_rate: float


@app.post("/config")
async def set_config(payload: ConfigPayload) -> dict:
    store.set_config(payload.key, BucketConfig(payload.capacity, payload.refill_rate))
    return {"updated": payload.key}


@app.get("/log")
async def replication_log() -> dict:
    return {"entries": store.replication_log()}
