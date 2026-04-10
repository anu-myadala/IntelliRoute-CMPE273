"""Central configuration for all IntelliRoute services.

Each service reads the ports and peer URLs from environment variables with
sensible defaults so the whole system can be brought up on a single host
without any configuration files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # Service ports
    gateway_port: int = _env_int("INTELLIROUTE_GATEWAY_PORT", 8000)
    router_port: int = _env_int("INTELLIROUTE_ROUTER_PORT", 8001)
    rate_limiter_port: int = _env_int("INTELLIROUTE_RATE_LIMITER_PORT", 8002)
    cost_tracker_port: int = _env_int("INTELLIROUTE_COST_TRACKER_PORT", 8003)
    health_monitor_port: int = _env_int("INTELLIROUTE_HEALTH_MONITOR_PORT", 8004)

    # Mock LLM providers (spun up on these ports for local dev / tests)
    mock_fast_port: int = _env_int("INTELLIROUTE_MOCK_FAST_PORT", 9001)
    mock_smart_port: int = _env_int("INTELLIROUTE_MOCK_SMART_PORT", 9002)
    mock_cheap_port: int = _env_int("INTELLIROUTE_MOCK_CHEAP_PORT", 9003)

    host: str = os.environ.get("INTELLIROUTE_HOST", "127.0.0.1")

    @property
    def router_url(self) -> str:
        return f"http://{self.host}:{self.router_port}"

    @property
    def rate_limiter_url(self) -> str:
        return f"http://{self.host}:{self.rate_limiter_port}"

    @property
    def cost_tracker_url(self) -> str:
        return f"http://{self.host}:{self.cost_tracker_port}"

    @property
    def health_monitor_url(self) -> str:
        return f"http://{self.host}:{self.health_monitor_port}"


settings = Settings()
