"""Rate limiter HTTP service with leader election.

Wraps the ``RateLimiterStore`` in a FastAPI app.  Gateway/router replicas
call ``/check`` on every outbound request.  Per-key configuration can be
updated at runtime via ``/config``.

Leader/follower semantics
-------------------------
Multiple replicas participate in a Bully leader election.  The leader
owns authoritative token-bucket state; followers forward ``/check``
requests to the leader.  If the leader fails, the highest-ID surviving
replica takes over via a new election.

Environment variables
---------------------
RATE_LIMITER_REPLICA_ID : str
    Unique ID for this replica (default ``"rl-0"``).
RATE_LIMITER_PEERS : str
    Comma-separated ``id=url`` pairs, e.g.
    ``"rl-1=http://127.0.0.1:8012,rl-2=http://127.0.0.1:8022"``.
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..common.logging import get_logger, log_event
from ..common.models import RateLimitCheck, RateLimitResult
from .election import ElectionConfig, ElectionState, LeaderElection, Peer
from .token_bucket import BucketConfig, RateLimiterStore

log = get_logger("rate_limiter")

# ------------------------------------------------------------------ config
_replica_id = os.environ.get("RATE_LIMITER_REPLICA_ID", "rl-0")
_peer_str = os.environ.get("RATE_LIMITER_PEERS", "")


def _parse_peers(raw: str) -> list[Peer]:
    """Parse ``"id=url,id=url"`` into a list of Peer objects."""
    peers: list[Peer] = []
    for part in raw.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        rid, url = part.split("=", 1)
        peers.append(Peer(replica_id=rid.strip(), url=url.strip()))
    return peers


_peers = _parse_peers(_peer_str)

# ---------------------------------------------------------------- globals
_default = BucketConfig(capacity=10, refill_rate=1.0)
store = RateLimiterStore(default_config=_default)
store.set_leader(_replica_id)  # initial: assume I am leader-like

election = LeaderElection(
    my_id=_replica_id,
    peers=_peers,
    config=ElectionConfig(
        election_timeout_s=2.0,
        heartbeat_interval_s=1.0,
        heartbeat_timeout_s=3.0,
    ),
)

app = FastAPI(title="IntelliRoute RateLimiter")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_http: Optional[httpx.AsyncClient] = None


# --------------------------------------------------------- lifecycle
@app.on_event("startup")
async def _startup() -> None:
    global _http
    _http = httpx.AsyncClient(timeout=2.0)
    # Kick off an initial election.
    asyncio.create_task(_run_election())
    # Background loops for heartbeats, watchdog, and log sync.
    asyncio.create_task(_heartbeat_loop())
    asyncio.create_task(_leader_watchdog())
    asyncio.create_task(_log_sync_loop())


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _http is not None:
        await _http.aclose()


# ---------------------------------------------------------- helpers

def _leader_url() -> Optional[str]:
    """Return the base URL of the current leader, or None if unknown/self."""
    leader_id = election.current_leader
    if leader_id is None or leader_id == _replica_id:
        return None
    for p in _peers:
        if p.replica_id == leader_id:
            return p.url
    return None


async def _forward_check(req: RateLimitCheck) -> RateLimitResult:
    """Forward a /check request to the current leader."""
    url = _leader_url()
    if url is None:
        # Fallback: process locally (leader unknown or self).
        return _local_check(req)
    assert _http is not None
    try:
        r = await _http.post(f"{url}/check", json=req.model_dump())
        if r.status_code == 200:
            return RateLimitResult(**r.json())
    except Exception as exc:
        log_event(log, "forward_check_failed", leader=url, error=str(exc))
    # Fail-open: if we can't reach the leader, allow locally.
    return _local_check(req)


def _local_check(req: RateLimitCheck) -> RateLimitResult:
    key = f"{req.tenant_id}|{req.provider}"
    allowed, remaining, retry_after = store.try_consume(key, amount=req.tokens_requested)
    return RateLimitResult(
        allowed=allowed,
        remaining=remaining,
        retry_after_ms=retry_after,
        leader_replica=election.current_leader or _replica_id,
    )


# ------------------------------------------------------- election RPCs

async def _run_election() -> None:
    """Execute the Bully election protocol over HTTP."""
    higher_ids = election.start_election()
    if not higher_ids:
        # I won immediately (highest ID).
        election.declare_victory()
        store.set_leader(_replica_id)
        log_event(log, "election_won", leader=_replica_id)
        await _broadcast_victory()
        return

    # Challenge all higher peers.
    assert _http is not None
    any_responded = False
    for pid in higher_ids:
        peer = next((p for p in _peers if p.replica_id == pid), None)
        if peer is None:
            continue
        try:
            r = await _http.post(
                f"{peer.url}/election/challenge",
                params={"from_id": _replica_id},
            )
            if r.status_code == 200 and r.json().get("alive"):
                any_responded = True
        except Exception:
            pass

    if not any_responded:
        # No higher peer responded — I win.
        election.declare_victory()
        store.set_leader(_replica_id)
        log_event(log, "election_won", leader=_replica_id)
        await _broadcast_victory()
    else:
        # A higher peer is alive; wait for its victory announcement.
        # The watchdog will re-trigger if we don't hear from a leader.
        await asyncio.sleep(election._config.election_timeout_s)
        if election.state == ElectionState.CANDIDATE:
            # Still no leader — retry.
            election.declare_victory()
            store.set_leader(_replica_id)
            log_event(log, "election_timeout_self_elect", leader=_replica_id)
            await _broadcast_victory()


async def _broadcast_victory() -> None:
    """Announce victory to all peers."""
    assert _http is not None
    for peer in _peers:
        try:
            await _http.post(
                f"{peer.url}/election/victory",
                params={"leader_id": _replica_id},
            )
        except Exception:
            pass


# ------------------------------------------------ background tasks

async def _heartbeat_loop() -> None:
    """If I am leader, send heartbeats to all peers periodically."""
    while True:
        await asyncio.sleep(election._config.heartbeat_interval_s)
        if election.state != ElectionState.LEADER:
            continue
        assert _http is not None
        for peer in _peers:
            try:
                await _http.post(
                    f"{peer.url}/election/heartbeat",
                    params={"from_leader": _replica_id},
                )
            except Exception:
                pass


async def _leader_watchdog() -> None:
    """If I am follower, detect leader timeout and start a new election."""
    while True:
        await asyncio.sleep(1.0)
        if election.check_leader_timeout():
            log_event(log, "leader_timeout_detected", old_leader=election.current_leader)
            await _run_election()


async def _log_sync_loop() -> None:
    """If I am follower, periodically pull new log entries from leader."""
    local_offset = 0
    while True:
        await asyncio.sleep(2.0)
        if election.is_leader:
            local_offset = store.log_length()
            continue
        url = _leader_url()
        if url is None:
            continue
        assert _http is not None
        try:
            r = await _http.get(f"{url}/log/since/{local_offset}")
            if r.status_code == 200:
                entries = r.json().get("entries", [])
                for entry in entries:
                    ts, key, amount, allowed = entry
                    store.replay_log_entry(ts, key, amount, allowed)
                    local_offset += 1
        except Exception:
            pass


# ------------------------------------------------------- HTTP endpoints

@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "replica_id": _replica_id,
        "leader": election.current_leader,
        "state": election.state.value,
    }


@app.get("/leader")
async def leader() -> dict:
    return {"leader": election.current_leader, "replica_id": _replica_id}


@app.post("/check", response_model=RateLimitResult)
async def check(req: RateLimitCheck) -> RateLimitResult:
    if election.is_leader:
        result = _local_check(req)
    else:
        result = await _forward_check(req)
    log_event(
        log,
        "rate_limit_check",
        key=f"{req.tenant_id}|{req.provider}",
        allowed=result.allowed,
        remaining=round(result.remaining, 3),
        retry_after_ms=result.retry_after_ms,
    )
    return result


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


@app.get("/log/since/{offset}")
async def replication_log_since(offset: int) -> dict:
    full_log = store.replication_log()
    return {"entries": full_log[offset:]}


# ------------------------------------------------ election endpoints

@app.post("/election/challenge")
async def election_challenge(from_id: str) -> dict:
    alive = election.receive_challenge(from_id)
    if alive:
        # I outrank the challenger; start my own election.
        asyncio.create_task(_run_election())
    return {"alive": alive}


@app.post("/election/victory")
async def election_victory(leader_id: str) -> dict:
    election.receive_victory(leader_id)
    store.set_leader(leader_id)
    log_event(log, "new_leader_accepted", leader=leader_id)
    return {"accepted": True}


@app.post("/election/heartbeat")
async def election_heartbeat(from_leader: str) -> dict:
    election.receive_heartbeat(from_leader)
    return {"ok": True}


@app.get("/election/status")
async def election_status() -> dict:
    return {
        "replica_id": _replica_id,
        "state": election.state.value,
        "current_leader": election.current_leader,
        "peers": [p.replica_id for p in _peers],
    }
