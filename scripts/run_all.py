"""Spin up every IntelliRoute service in one process.

This is the simplest way to run the full system locally. Each service
listens on its configured port and talks to the others via HTTP.

Usage:
    python scripts/run_all.py

The script blocks until Ctrl-C.
"""
from __future__ import annotations

import asyncio
import os
import sys

import uvicorn

# Make the sibling package importable when running from source.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from intelliroute.common.config import settings


def _provider_env(name: str, model: str, latency: float, failure: float, cost: float) -> dict:
    return {
        "MOCK_NAME": name,
        "MOCK_MODEL": model,
        "MOCK_LATENCY_MS": str(latency),
        "MOCK_LATENCY_JITTER_MS": str(latency * 0.2),
        "MOCK_FAILURE_RATE": str(failure),
        "MOCK_COST_PER_1K": str(cost),
    }


def _make_config(module: str, port: int, env: dict | None = None) -> uvicorn.Config:
    if env:
        for k, v in env.items():
            os.environ[k] = v
    return uvicorn.Config(module, host=settings.host, port=port, log_level="warning")


async def _serve(cfg: uvicorn.Config) -> None:
    server = uvicorn.Server(cfg)
    await server.serve()


async def main() -> None:
    # NOTE: in a single process, env vars for the three mock providers
    # would collide if applied all at once. Instead we spawn them as
    # separate subprocess-less uvicorn apps by configuring each app at
    # startup via import-time env lookups. See mock_provider.main.
    # For the single-process runner we therefore launch three separate
    # Python processes for the mock providers in practice; to keep this
    # script simple and dependency-free we just run the core services
    # here and ask the user to run each mock provider with its own env.
    configs = [
        _make_config("intelliroute.rate_limiter.main:app", settings.rate_limiter_port),
        _make_config("intelliroute.cost_tracker.main:app", settings.cost_tracker_port),
        _make_config("intelliroute.health_monitor.main:app", settings.health_monitor_port),
        _make_config("intelliroute.router.main:app", settings.router_port),
        _make_config("intelliroute.gateway.main:app", settings.gateway_port),
    ]
    await asyncio.gather(*(_serve(c) for c in configs))


if __name__ == "__main__":
    asyncio.run(main())
