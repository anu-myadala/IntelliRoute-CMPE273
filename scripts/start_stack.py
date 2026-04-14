"""Launch the entire IntelliRoute stack as subprocesses.

Each service runs in its own ``python -m uvicorn`` subprocess so the
three mock providers can read their distinct env vars cleanly. Press
Ctrl-C to stop all services.

Usage:
    python scripts/start_stack.py
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


SERVICES = [
    {
        "module": "intelliroute.mock_provider.main:app",
        "port_env": "INTELLIROUTE_MOCK_FAST_PORT",
        "default_port": 9001,
        "env": {
            "MOCK_NAME": "mock-fast", "MOCK_MODEL": "fast-1",
            "MOCK_LATENCY_MS": "30", "MOCK_LATENCY_JITTER_MS": "10",
            "MOCK_FAILURE_RATE": "0.0", "MOCK_COST_PER_1K": "0.002",
        },
    },
    {
        "module": "intelliroute.mock_provider.main:app",
        "port_env": "INTELLIROUTE_MOCK_SMART_PORT",
        "default_port": 9002,
        "env": {
            "MOCK_NAME": "mock-smart", "MOCK_MODEL": "smart-1",
            "MOCK_LATENCY_MS": "120", "MOCK_LATENCY_JITTER_MS": "20",
            "MOCK_FAILURE_RATE": "0.0", "MOCK_COST_PER_1K": "0.02",
        },
    },
    {
        "module": "intelliroute.mock_provider.main:app",
        "port_env": "INTELLIROUTE_MOCK_CHEAP_PORT",
        "default_port": 9003,
        "env": {
            "MOCK_NAME": "mock-cheap", "MOCK_MODEL": "cheap-1",
            "MOCK_LATENCY_MS": "80", "MOCK_LATENCY_JITTER_MS": "15",
            "MOCK_FAILURE_RATE": "0.0", "MOCK_COST_PER_1K": "0.0003",
        },
    },
    {
        "module": "intelliroute.rate_limiter.main:app",
        "port_env": "INTELLIROUTE_RATE_LIMITER_PORT",
        "default_port": 8002,
        "env": {
            "RATE_LIMITER_REPLICA_ID": "rl-0",
            "RATE_LIMITER_PEERS": "rl-1=http://127.0.0.1:8012,rl-2=http://127.0.0.1:8022",
        },
    },
    {
        "module": "intelliroute.rate_limiter.main:app",
        "port_env": "_RL_REPLICA_1_PORT",
        "default_port": 8012,
        "env": {
            "RATE_LIMITER_REPLICA_ID": "rl-1",
            "RATE_LIMITER_PEERS": "rl-0=http://127.0.0.1:8002,rl-2=http://127.0.0.1:8022",
        },
    },
    {
        "module": "intelliroute.rate_limiter.main:app",
        "port_env": "_RL_REPLICA_2_PORT",
        "default_port": 8022,
        "env": {
            "RATE_LIMITER_REPLICA_ID": "rl-2",
            "RATE_LIMITER_PEERS": "rl-0=http://127.0.0.1:8002,rl-1=http://127.0.0.1:8012",
        },
    },
    {"module": "intelliroute.cost_tracker.main:app",   "port_env": "INTELLIROUTE_COST_TRACKER_PORT",   "default_port": 8003, "env": {}},
    {"module": "intelliroute.health_monitor.main:app", "port_env": "INTELLIROUTE_HEALTH_MONITOR_PORT", "default_port": 8004, "env": {}},
    {"module": "intelliroute.router.main:app",         "port_env": "INTELLIROUTE_ROUTER_PORT",         "default_port": 8001, "env": {}},
    {"module": "intelliroute.gateway.main:app",        "port_env": "INTELLIROUTE_GATEWAY_PORT",        "default_port": 8000, "env": {}},
]


def main() -> int:
    procs: list[subprocess.Popen] = []
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(ROOT)

    try:
        for svc in SERVICES:
            port = int(os.environ.get(svc["port_env"], svc["default_port"]))
            env = base_env.copy()
            env.update(svc["env"])
            cmd = [
                sys.executable, "-m", "uvicorn", svc["module"],
                "--host", os.environ.get("INTELLIROUTE_HOST", "127.0.0.1"),
                "--port", str(port),
                "--log-level", "info",
            ]
            print(f"starting {svc['module']} on :{port}")
            procs.append(subprocess.Popen(cmd, env=env, cwd=str(ROOT)))
            time.sleep(0.15)

        # Start a simple HTTP server for the frontend UI
        frontend_port = int(os.environ.get("INTELLIROUTE_FRONTEND_PORT", "3000"))
        frontend_dir = str(ROOT / "frontend")
        if os.path.isdir(frontend_dir):
            frontend_cmd = [
                sys.executable, "-m", "http.server",
                str(frontend_port), "--directory", frontend_dir,
                "--bind", os.environ.get("INTELLIROUTE_HOST", "127.0.0.1"),
            ]
            print(f"starting frontend on :{frontend_port}")
            procs.append(subprocess.Popen(frontend_cmd, env=base_env, cwd=str(ROOT)))

        print(f"\nIntelliRoute stack running.")
        print(f"  Gateway:    http://127.0.0.1:8000")
        print(f"  Frontend:   http://127.0.0.1:{frontend_port}")
        print(f"  Press Ctrl-C to stop.\n")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nshutting down...")
    finally:
        for p in procs:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
