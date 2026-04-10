"""Demo client.

Hits the running gateway with a few representative requests, prints the
routing decisions, then prints the tenant cost summary at the end.

Usage:
    # Terminal 1: launch the stack (or use docker compose / start.sh)
    python scripts/start_stack.py
    # Terminal 2:
    python scripts/demo.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import httpx

GATEWAY = os.environ.get("INTELLIROUTE_DEMO_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("INTELLIROUTE_DEMO_KEY", "demo-key-123")


def _post(path: str, body: dict) -> dict:
    r = httpx.post(
        f"{GATEWAY}{path}", json=body,
        headers={"X-API-Key": KEY, "Content-Type": "application/json"},
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
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
            print(f"[{label}] FAILED: {exc}")
            return 1
        print(f"[{label}] -> {resp['provider']} ({resp['model']}) "
              f"latency={resp['latency_ms']}ms cost=${resp['estimated_cost_usd']:.6f}"
              f" fallback={resp['fallback_used']}")

    time.sleep(0.3)  # let async cost events flush
    summary = httpx.get(
        f"{GATEWAY}/v1/cost/summary", headers={"X-API-Key": KEY}, timeout=5.0
    ).json()
    print("\nCost summary:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
