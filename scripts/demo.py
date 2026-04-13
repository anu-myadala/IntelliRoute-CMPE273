"""Demo client.

Hits the running gateway with a few representative requests, prints the
routing decisions, then demonstrates the new features: feedback loop,
backpressure / load shedding, and leader election status.

Usage:
    # Terminal 1: launch the stack (or use docker compose / start.sh)
    python scripts/start_stack.py
    # Terminal 2:
    python scripts/demo.py
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time

import httpx

GATEWAY = os.environ.get("INTELLIROUTE_DEMO_URL", "http://127.0.0.1:8000")
ROUTER = os.environ.get("INTELLIROUTE_ROUTER_URL", "http://127.0.0.1:8001")
KEY = os.environ.get("INTELLIROUTE_DEMO_KEY", "demo-key-123")

RL_REPLICAS = [
    os.environ.get("RL_0_URL", "http://127.0.0.1:8002"),
    os.environ.get("RL_1_URL", "http://127.0.0.1:8012"),
    os.environ.get("RL_2_URL", "http://127.0.0.1:8022"),
]


def _post(path: str, body: dict) -> dict:
    r = httpx.post(
        f"{GATEWAY}{path}", json=body,
        headers={"X-API-Key": KEY, "Content-Type": "application/json"},
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()


def demo_routing() -> int:
    """Original demo: send typed requests and observe routing decisions."""
    print("=" * 60)
    print("DEMO 1: Intent-Aware Routing")
    print("=" * 60)

    prompts = [
        ("interactive small-talk", "Hi, what's the capital of France?"),
        ("reasoning", (
            "Explain step by step why the CAP theorem implies that a "
            "distributed system cannot simultaneously offer consistency, "
            "availability, and partition tolerance, and analyze how real "
            "systems trade off these properties."
        )),
        ("batch summarisation", "Summarize the following document into bullet points: ..."),
        ("code", "I got an exception in my Python loop, here is the traceback ..."),
    ]
    for label, text in prompts:
        body = {
            "tenant_id": "ignored",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 80,
        }
        try:
            resp = _post("/v1/complete", body)
        except Exception as exc:
            print(f"  [{label}] FAILED: {exc}")
            return 1
        print(f"  [{label}] -> {resp['provider']} ({resp['model']}) "
              f"latency={resp['latency_ms']}ms cost=${resp['estimated_cost_usd']:.6f}"
              f" fallback={resp['fallback_used']}")

    time.sleep(0.3)  # let async cost events flush
    summary = httpx.get(
        f"{GATEWAY}/v1/cost/summary", headers={"X-API-Key": KEY}, timeout=5.0
    ).json()
    print("\n  Cost summary:")
    print(f"    requests={summary['total_requests']}  "
          f"tokens={summary['total_tokens']}  "
          f"cost=${summary['total_cost_usd']:.6f}")
    print(f"    by_provider={json.dumps(summary['by_provider'])}")
    return 0


def demo_feedback() -> None:
    """Query the feedback endpoint to show learned EMA metrics."""
    print("\n" + "=" * 60)
    print("DEMO 2: Feedback Loop (Continuous Learning)")
    print("=" * 60)

    try:
        r = httpx.get(f"{ROUTER}/feedback", timeout=5.0)
        data = r.json()
        metrics = data.get("metrics", {})
        if not metrics:
            print("  No feedback metrics yet (send more requests first).")
            return
        for provider, m in metrics.items():
            print(f"  {provider}:")
            print(f"    latency_ema    = {m['latency_ema']:.1f}ms")
            print(f"    success_rate   = {m['success_rate_ema']:.3f}")
            print(f"    efficiency     = {m['token_efficiency_ema']:.3f}")
            print(f"    anomaly_score  = {m['anomaly_score']:.3f}")
            print(f"    samples        = {m['sample_count']}")
    except Exception as exc:
        print(f"  Failed to query feedback: {exc}")


def demo_backpressure() -> None:
    """Send a burst of concurrent batch requests to show load shedding."""
    print("\n" + "=" * 60)
    print("DEMO 3: Backpressure & Load Shedding")
    print("=" * 60)

    # First show queue stats baseline
    try:
        r = httpx.get(f"{ROUTER}/queue/stats", timeout=5.0)
        stats = r.json()
        print(f"  Queue stats (before): depth={stats['total_depth']} "
              f"shed={stats['shed_count']} timeouts={stats['timeout_count']}")
    except Exception:
        pass

    # Send a burst of batch requests concurrently
    batch_body = {
        "tenant_id": "ignored",
        "messages": [{"role": "user", "content": "Summarize the following into bullet points: ..."}],
        "max_tokens": 50,
    }

    print("  Sending 10 concurrent batch requests...")
    successes = 0
    shed = 0

    def _send_batch():
        try:
            r = httpx.post(
                f"{GATEWAY}/v1/complete", json=batch_body,
                headers={"X-API-Key": KEY}, timeout=15.0,
            )
            return r.status_code
        except Exception:
            return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_send_batch) for _ in range(10)]
        for f in concurrent.futures.as_completed(futures):
            code = f.result()
            if code == 200:
                successes += 1
            elif code == 429:
                shed += 1

    print(f"  Results: {successes} succeeded, {shed} shed (429)")

    # Show updated queue stats
    try:
        r = httpx.get(f"{ROUTER}/queue/stats", timeout=5.0)
        stats = r.json()
        print(f"  Queue stats (after):  depth={stats['total_depth']} "
              f"shed={stats['shed_count']} timeouts={stats['timeout_count']}")
    except Exception:
        pass

    # Show that interactive requests still sail through
    interactive_body = {
        "tenant_id": "ignored",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 30,
    }
    try:
        resp = _post("/v1/complete", interactive_body)
        print(f"  Interactive request (bypasses queue): "
              f"provider={resp['provider']} latency={resp['latency_ms']}ms")
    except Exception as exc:
        print(f"  Interactive request failed: {exc}")


def demo_election() -> None:
    """Query all rate limiter replicas to show election state."""
    print("\n" + "=" * 60)
    print("DEMO 4: Leader Election (Distributed Coordination)")
    print("=" * 60)

    for url in RL_REPLICAS:
        try:
            r = httpx.get(f"{url}/election/status", timeout=2.0)
            if r.status_code == 200:
                data = r.json()
                marker = " <-- LEADER" if data["state"] == "leader" else ""
                print(f"  {data['replica_id']}: state={data['state']}  "
                      f"leader={data['current_leader']}  "
                      f"peers={data['peers']}{marker}")
            else:
                print(f"  {url}: HTTP {r.status_code}")
        except Exception:
            print(f"  {url}: unreachable")


def main() -> int:
    rc = demo_routing()
    if rc != 0:
        return rc
    demo_feedback()
    demo_backpressure()
    demo_election()
    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
