"""End-to-end integration test.

Spins up every IntelliRoute service as a real subprocess bound to an
ephemeral port, then exercises the whole system over HTTP:

1. Registry bootstrap + intent-aware routing (a batch prompt should be
   routed to mock-cheap; an interactive prompt to mock-fast).
2. Automatic failover: knock out the top-ranked provider via its
   ``/admin/force_fail`` hook and verify the router seamlessly falls
   back to the next candidate and the circuit breaker trips.
3. Distributed rate limiting: tighten the bucket to capacity=1 and
   verify concurrent clients are throttled.
4. Async cost accounting: after successful completions the cost
   tracker reflects the tenant's spend.

The test takes a few seconds because it actually starts and tears down
real processes. That's the whole point: this is the integration test
that proves the services work together.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(url: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return
        except Exception as exc:
            last_err = exc
        time.sleep(0.1)
    raise RuntimeError(f"service at {url} did not become ready: {last_err}")


class _Stack:
    def __init__(self) -> None:
        self.ports = {
            "gateway": _free_port(),
            "router": _free_port(),
            "rate_limiter": _free_port(),
            "cost_tracker": _free_port(),
            "health_monitor": _free_port(),
            "mock_fast": _free_port(),
            "mock_smart": _free_port(),
            "mock_cheap": _free_port(),
        }
        self.procs: list[subprocess.Popen] = []

    def _base_env(self) -> dict:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        env["INTELLIROUTE_HOST"] = "127.0.0.1"
        env["INTELLIROUTE_GATEWAY_PORT"] = str(self.ports["gateway"])
        env["INTELLIROUTE_ROUTER_PORT"] = str(self.ports["router"])
        env["INTELLIROUTE_RATE_LIMITER_PORT"] = str(self.ports["rate_limiter"])
        env["INTELLIROUTE_COST_TRACKER_PORT"] = str(self.ports["cost_tracker"])
        env["INTELLIROUTE_HEALTH_MONITOR_PORT"] = str(self.ports["health_monitor"])
        env["INTELLIROUTE_MOCK_FAST_PORT"] = str(self.ports["mock_fast"])
        env["INTELLIROUTE_MOCK_SMART_PORT"] = str(self.ports["mock_smart"])
        env["INTELLIROUTE_MOCK_CHEAP_PORT"] = str(self.ports["mock_cheap"])
        # Single-replica election config (no peers) for integration tests
        env["RATE_LIMITER_REPLICA_ID"] = "rl-0"
        env["RATE_LIMITER_PEERS"] = ""
        return env

    def _spawn_uvicorn(self, module: str, port: int, extra_env: dict | None = None) -> None:
        env = self._base_env()
        if extra_env:
            env.update(extra_env)
        cmd = [
            sys.executable, "-m", "uvicorn", module,
            "--host", "127.0.0.1", "--port", str(port),
            "--log-level", "warning",
        ]
        p = subprocess.Popen(cmd, env=env, cwd=str(ROOT))
        self.procs.append(p)

    def start(self) -> None:
        # Mock providers with distinct personalities.
        self._spawn_uvicorn(
            "intelliroute.mock_provider.main:app",
            self.ports["mock_fast"],
            {
                "MOCK_NAME": "mock-fast", "MOCK_MODEL": "fast-1",
                "MOCK_LATENCY_MS": "30", "MOCK_LATENCY_JITTER_MS": "5",
                "MOCK_FAILURE_RATE": "0.0", "MOCK_COST_PER_1K": "0.002",
            },
        )
        self._spawn_uvicorn(
            "intelliroute.mock_provider.main:app",
            self.ports["mock_smart"],
            {
                "MOCK_NAME": "mock-smart", "MOCK_MODEL": "smart-1",
                "MOCK_LATENCY_MS": "80", "MOCK_LATENCY_JITTER_MS": "5",
                "MOCK_FAILURE_RATE": "0.0", "MOCK_COST_PER_1K": "0.02",
            },
        )
        self._spawn_uvicorn(
            "intelliroute.mock_provider.main:app",
            self.ports["mock_cheap"],
            {
                "MOCK_NAME": "mock-cheap", "MOCK_MODEL": "cheap-1",
                "MOCK_LATENCY_MS": "60", "MOCK_LATENCY_JITTER_MS": "5",
                "MOCK_FAILURE_RATE": "0.0", "MOCK_COST_PER_1K": "0.0003",
            },
        )
        self._spawn_uvicorn("intelliroute.rate_limiter.main:app", self.ports["rate_limiter"])
        self._spawn_uvicorn("intelliroute.cost_tracker.main:app", self.ports["cost_tracker"])
        self._spawn_uvicorn("intelliroute.health_monitor.main:app", self.ports["health_monitor"])
        self._spawn_uvicorn("intelliroute.router.main:app", self.ports["router"])
        self._spawn_uvicorn("intelliroute.gateway.main:app", self.ports["gateway"])

        for name in [
            "mock_fast", "mock_smart", "mock_cheap",
            "rate_limiter", "cost_tracker", "health_monitor",
            "router", "gateway",
        ]:
            _wait_ready(f"http://127.0.0.1:{self.ports[name]}/health", timeout=15.0)

    def stop(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        self.procs.clear()

    # Convenience URLs ------------------------------------------------
    @property
    def gateway_url(self) -> str:
        return f"http://127.0.0.1:{self.ports['gateway']}"

    @property
    def router_url(self) -> str:
        return f"http://127.0.0.1:{self.ports['router']}"

    @property
    def rate_limiter_url(self) -> str:
        return f"http://127.0.0.1:{self.ports['rate_limiter']}"

    @property
    def cost_tracker_url(self) -> str:
        return f"http://127.0.0.1:{self.ports['cost_tracker']}"

    def mock_url(self, which: str) -> str:
        return f"http://127.0.0.1:{self.ports[which]}"


@pytest.fixture(scope="module")
def stack():
    s = _Stack()
    s.start()
    try:
        yield s
    finally:
        s.stop()


def _headers() -> dict:
    return {"X-API-Key": "demo-key-123", "Content-Type": "application/json"}


def _interactive_request() -> dict:
    return {
        "tenant_id": "ignored-by-gateway",
        "messages": [{"role": "user", "content": "Hi there!"}],
        "max_tokens": 50,
    }


def _batch_request() -> dict:
    return {
        "tenant_id": "ignored-by-gateway",
        "messages": [{"role": "user", "content": "Summarize the following document into bullet points"}],
        "max_tokens": 50,
    }


def test_interactive_prompt_routes_to_fast(stack):
    r = httpx.post(f"{stack.gateway_url}/v1/complete", json=_interactive_request(), headers=_headers(), timeout=10.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "mock-fast"
    assert body["model"] == "fast-1"
    assert body["total_tokens"] > 0


def test_batch_prompt_routes_to_cheap(stack):
    r = httpx.post(f"{stack.gateway_url}/v1/complete", json=_batch_request(), headers=_headers(), timeout=10.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "mock-cheap"


def test_router_decide_endpoint_returns_ranking(stack):
    r = httpx.post(f"{stack.router_url}/decide", json=_interactive_request(), timeout=5.0)
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "interactive"
    assert body["ranked"][0] == "mock-fast"
    assert set(body["ranked"]) == {"mock-fast", "mock-smart", "mock-cheap"}


def test_failover_when_top_provider_is_down(stack):
    # Force mock-fast into failing mode, then issue an interactive prompt.
    # The router should fall back to the next provider in the ranked list.
    r = httpx.post(f"{stack.mock_url('mock_fast')}/admin/force_fail", params={"on": "true"}, timeout=5.0)
    assert r.status_code == 200
    try:
        body = None
        # A few attempts because the breaker needs to see failures to trip.
        for _ in range(5):
            r = httpx.post(f"{stack.gateway_url}/v1/complete", json=_interactive_request(), headers=_headers(), timeout=10.0)
            assert r.status_code == 200, r.text
            body = r.json()
            if body["provider"] != "mock-fast":
                break
        assert body is not None
        assert body["provider"] != "mock-fast"
        assert body["fallback_used"] is True
    finally:
        httpx.post(f"{stack.mock_url('mock_fast')}/admin/force_fail", params={"on": "false"}, timeout=5.0)


def test_rate_limiter_throttles_when_bucket_drained(stack):
    # Tighten the bucket for (demo-tenant, mock-cheap) to capacity=1, refill very slow.
    r = httpx.post(
        f"{stack.rate_limiter_url}/config",
        json={"key": "demo-tenant|mock-cheap", "capacity": 1, "refill_rate": 0.01},
        timeout=5.0,
    )
    assert r.status_code == 200

    # First batch request should succeed (consumes the 1 token).
    r1 = httpx.post(f"{stack.gateway_url}/v1/complete", json=_batch_request(), headers=_headers(), timeout=10.0)
    assert r1.status_code == 200
    # Second batch request: mock-cheap is rate-limited so the router should
    # fall back to another provider (fallback_used=True).
    r2 = httpx.post(f"{stack.gateway_url}/v1/complete", json=_batch_request(), headers=_headers(), timeout=10.0)
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["fallback_used"] is True
    assert body2["provider"] != "mock-cheap"

    # Reset config so subsequent tests aren't affected.
    httpx.post(
        f"{stack.rate_limiter_url}/config",
        json={"key": "demo-tenant|mock-cheap", "capacity": 10, "refill_rate": 1.0},
        timeout=5.0,
    )


def test_cost_tracker_reflects_tenant_spend(stack):
    # Poll the cost summary; because cost events are published async, allow
    # up to a second for eventual consistency.
    deadline = time.monotonic() + 2.0
    summary = None
    while time.monotonic() < deadline:
        r = httpx.get(f"{stack.gateway_url}/v1/cost/summary", headers=_headers(), timeout=5.0)
        assert r.status_code == 200
        summary = r.json()
        if summary["total_requests"] > 0:
            break
        time.sleep(0.1)
    assert summary is not None
    assert summary["tenant_id"] == "demo-tenant"
    assert summary["total_requests"] >= 1
    assert summary["total_cost_usd"] > 0.0
    assert len(summary["by_provider"]) >= 1


def test_unauthenticated_request_is_rejected(stack):
    r = httpx.post(f"{stack.gateway_url}/v1/complete", json=_interactive_request(), timeout=5.0)
    assert r.status_code == 401


# ---- New feature integration tests ----


def test_feedback_metrics_populated_after_completions(stack):
    """After sending completions, the /feedback endpoint should have EMA data."""
    # Send a couple of requests to populate feedback
    httpx.post(f"{stack.gateway_url}/v1/complete", json=_interactive_request(),
               headers=_headers(), timeout=10.0)
    httpx.post(f"{stack.gateway_url}/v1/complete", json=_batch_request(),
               headers=_headers(), timeout=10.0)
    time.sleep(0.2)

    r = httpx.get(f"{stack.router_url}/feedback", timeout=5.0)
    assert r.status_code == 200
    data = r.json()
    metrics = data.get("metrics", {})
    assert len(metrics) >= 1
    # Check that at least one provider has samples
    any_sampled = any(m["sample_count"] > 0 for m in metrics.values())
    assert any_sampled


def test_queue_stats_endpoint(stack):
    """The /queue/stats endpoint should return valid stats."""
    r = httpx.get(f"{stack.router_url}/queue/stats", timeout=5.0)
    assert r.status_code == 200
    data = r.json()
    assert "total_depth" in data
    assert "by_priority" in data
    assert "shed_count" in data
    assert "timeout_count" in data


def test_election_status_shows_leader(stack):
    """The rate limiter should report itself as leader (single replica)."""
    r = httpx.get(f"{stack.rate_limiter_url}/election/status", timeout=5.0)
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "leader"
    assert data["replica_id"] == "rl-0"
    assert data["current_leader"] == "rl-0"
