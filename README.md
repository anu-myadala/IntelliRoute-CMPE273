# IntelliRoute

**A distributed control plane for multi-LLM orchestration.**

CMPE 273 — Enterprise Distributed Systems, Spring 2026, San José State University.

**Team:** Anukrithi Myadala, Larry Nguyen, James Pham, Surbhi Singh.

IntelliRoute is a multi-service distributed system that sits between
applications and LLM providers (OpenAI, Anthropic, local open-source
models, …) and turns them into a single, governed compute resource. It
classifies each request by intent, picks the best provider with a
multi-objective scoring policy, enforces distributed rate limits,
fails over automatically when providers degrade, and tracks cost per
tenant — all while exposing the internal state needed for observability.

---

## Table of contents

1. [Why this exists](#why-this-exists)
2. [What it does](#what-it-does)
3. [Architecture](#architecture)
4. [Distributed-systems concepts demonstrated](#distributed-systems-concepts-demonstrated)
5. [**Quick start for teammates** (setup, run, test, demo)](#quick-start-for-teammates)
6. [Project layout](#project-layout)
7. [API surface](#api-surface)
8. [Configuration](#configuration)
9. [Team contributions](#team-contributions)
10. [What is intentionally simplified](#what-is-intentionally-simplified)

---

## Why this exists

Organisations integrating multiple LLM providers face a familiar set of
distributed-systems problems:

- **Unpredictable latency** across providers and regions.
- **Fragmented rate limits and quotas** that have to be enforced across
  many client services.
- **Escalating, opaque costs.**
- **No unified place to make routing decisions** that balance cost,
  latency, accuracy, and availability.
- **No system-level optimisation** across heterogeneous workloads
  (interactive, reasoning, batch, code).

IntelliRoute addresses these by treating LLMs as **distributed compute
resources behind a control plane**, not as isolated APIs.

## What it does

1. **Intent-aware routing.** Each request is classified into one of
   `interactive`, `reasoning`, `batch`, or `code`. The routing policy
   picks the best provider for that intent.
2. **Multi-objective scoring.** Providers are ranked by a weighted
   combination of latency, cost, capability, and historical success
   rate. Weights vary per intent.
3. **Distributed rate limiting.** A token-bucket service enforces
   per-tenant, per-provider quotas. The bucket store is the
   authoritative leader; followers replicate from a replication log.
4. **Adaptive fallback.** If the primary provider is rate-limited or
   the circuit breaker is open, the router automatically tries the
   next-best provider. If none are healthy, it returns a structured
   error rather than blocking.
5. **Cost-aware accounting.** Cost events are published asynchronously
   to a tracker that maintains per-tenant rollups and fires
   budget-exceeded alerts.
6. **Observability surface.** Every service emits structured JSON logs;
   the gateway exposes aggregate health and cost views.

## Architecture

```
                ┌──────────────────────────────────────────────────┐
                │                                                  │
   client ──▶  Gateway  ──sync(HTTP)───────▶  Router  ──▶  Mock LLM Providers
   (X-API-Key)   :8000                        :8001            :9001 mock-fast
                                  │                             :9002 mock-smart
                                  │                             :9003 mock-cheap
                                  ├──sync──▶  Rate Limiter  :8002
                                  │              (leader + replication log)
                                  ├──sync──▶  Health Monitor :8004
                                  │              (circuit breakers + polling)
                                  └──async─▶  Cost Tracker  :8003
                                                 (per-tenant rollups + alerts)
```

Six service types, all written in Python with FastAPI:

| Service        | Port      | Responsibility                                                    |
|----------------|-----------|-------------------------------------------------------------------|
| Gateway        | 8000      | Public entry point, API-key auth, request tracing                 |
| Router         | 8001      | Intent classification, multi-objective ranking, fallback control  |
| Rate Limiter   | 8002      | Distributed token-bucket, replication log                         |
| Cost Tracker   | 8003      | Async cost events, per-tenant rollups, budget alerts              |
| Health Monitor | 8004      | Per-provider circuit breakers, periodic liveness polling          |
| Mock Providers | 9001-9003 | Three simulated upstream LLMs with distinct latency/cost profiles |

## Distributed-systems concepts demonstrated

This project intentionally maps to the major themes of CMPE 273:

| Course topic                       | Where it lives in IntelliRoute                                                                                          |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| Service discovery / naming         | `intelliroute/router/registry.py` — `ProviderRegistry` is the in-memory analogue of Consul/etcd                         |
| Communication: sync + async        | HTTP between gateway/router/limiter; fire-and-forget async cost events to the cost tracker                              |
| Coordination & leader election     | `RateLimiterStore` exposes a `leader_id` and a replication log; followers tail the log and replay state                 |
| Consistency & replication          | Strong consistency for rate-limit decisions (single leader); eventual consistency for cost rollups                      |
| Fault tolerance & graceful degrade | Circuit breakers per provider (`health_monitor/circuit_breaker.py`) plus router fallback chain                          |
| Backpressure & queueing            | The rate limiter is the explicit backpressure mechanism; the router treats throttle responses as a fallback signal     |
| Multi-objective optimisation       | `router/policy.py` — providers ranked on (latency, cost, capability, success) with intent-specific weights              |
| Security                           | API-key auth at the gateway; tenant identity is rewritten from the authenticated principal, not from the request body  |
| Observability                      | Structured JSON logger (`common/logging.py`), `/snapshot` endpoints, X-Request-Id propagation through the gateway      |

---

## Quick start for teammates

**Target audience:** Anukrithi, Larry, James, Surbhi. Follow these exact
steps on a fresh clone and you will have a running, fully-tested
IntelliRoute stack in under 5 minutes. Tested on macOS and Linux with
Python 3.10+.

### Prerequisites

- **Python 3.10 or newer.** Check with `python3 --version`.
- **pip** (comes with Python).
- Ports **8000-8004** and **9001-9003** free on localhost.

### Step 1 — Get the code

Unzip the project (or `git clone` it) and `cd` into it:

```bash
cd intelliroute
```

You should see `README.md`, `pyproject.toml`, and the folders
`intelliroute/`, `tests/`, and `scripts/`.

### Step 2 — Install dependencies

We recommend a virtualenv so you don't pollute system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" httpx pydantic pytest pytest-asyncio
```

That is the complete dependency list. There are no build steps, no
compiled extensions, and no external databases.

### Step 3 — Run the full test suite

This is the fastest way to confirm your environment is working. The
suite has **42 tests**: 35 unit tests and 7 integration tests that
actually spawn the entire 8-process stack and hit it over HTTP.

From the `intelliroute/` directory:

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
```

Expected outcome:

```
============================== 42 passed in ~2s ==============================
```

If anything fails, re-read the prerequisites (likely Python version or
a port collision). You can run just the pure unit tests (no
subprocesses) with:

```bash
PYTHONPATH=. python3 -m pytest tests/ -v --ignore=tests/test_integration.py
```

### Step 4 — Launch the system

Open **two terminals**.

**Terminal 1 — start the stack:**

```bash
cd intelliroute
source .venv/bin/activate
PYTHONPATH=. python3 scripts/start_stack.py
```

You should see ten "starting ..." lines (gateway, router, 3 rate
limiter replicas, health monitor, cost tracker, 3 mock providers) plus
a frontend server, followed by:

```
IntelliRoute stack running.
  Gateway:    http://127.0.0.1:8000
  Frontend:   http://127.0.0.1:3000
  Press Ctrl-C to stop.
```

Open **http://127.0.0.1:3000** in a browser to use the **live demo
dashboard** — a dark-themed UI with a chat interface, provider health
cards, routing visualization, cost tracker, and leader election status.
The dashboard polls all backend services every 2 seconds and displays
real-time metrics. It also has buttons to force-fail a provider, recover
it, and run a 20-request burst test.

**Terminal 2 — run the demo client (optional CLI alternative):**

```bash
cd intelliroute
source .venv/bin/activate
PYTHONPATH=. python3 scripts/demo.py
```

Expected output (numbers will vary by a few ms):

```
[interactive small-talk] -> mock-fast (fast-1)  latency=35ms  cost=$0.000024 fallback=False
[reasoning]              -> mock-smart (smart-1) latency=138ms cost=$0.001560 fallback=False
[batch summarisation]    -> mock-cheap (cheap-1) latency=95ms  cost=$0.000006 fallback=False
[code]                   -> mock-fast (fast-1)  latency=26ms  cost=$0.000044 fallback=False

Cost summary:
{
  "tenant_id": "demo-tenant",
  "total_requests": 4,
  "total_tokens": 133,
  "total_cost_usd": 0.001634,
  ...
}
```

**What this proves:**

- Intent classification: short chat → `interactive`, long "explain step
  by step" → `reasoning`, "summarize the following" → `batch`,
  traceback keyword → `code`.
- Routing policy: interactive and code pick `mock-fast` (low latency),
  reasoning picks `mock-smart` (highest capability for reasoning), and
  batch picks `mock-cheap` (lowest cost).
- Async cost tracker: the summary query hits a different service
  (port 8003) and reflects all 4 requests.

### Step 5 — Demo the failure modes (for the class presentation)

With the stack still running in Terminal 1, exercise each
distributed-systems concept live in Terminal 2.

**(a) Routing introspection — see the policy ranking without executing:**

```bash
curl -s -X POST http://127.0.0.1:8001/decide \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"demo-tenant","messages":[{"role":"user","content":"Explain step by step the CAP theorem and analyze the tradeoffs in practice, with enough context to be considered reasoning."}]}' | python3 -m json.tool
```

You should see `"intent": "reasoning"` and `"mock-smart"` first in the ranking.

**(b) Automatic failover — knock out the top provider:**

```bash
# Force mock-fast into failing mode
curl -s -X POST "http://127.0.0.1:9001/admin/force_fail?on=true"

# Send an interactive request - router should fall back to another provider
curl -s -X POST http://127.0.0.1:8000/v1/complete \
  -H "X-API-Key: demo-key-123" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"x","messages":[{"role":"user","content":"hi"}]}' | python3 -m json.tool

# Look at the breaker state - after a few failures it should be 'open'
curl -s http://127.0.0.1:8004/snapshot | python3 -m json.tool

# Restore
curl -s -X POST "http://127.0.0.1:9001/admin/force_fail?on=false"
```

**(c) Distributed rate limiting — tighten the bucket at runtime:**

```bash
# Shrink the bucket for (demo-tenant, mock-cheap) to 1 token, glacial refill
curl -s -X POST http://127.0.0.1:8002/config \
  -H "Content-Type: application/json" \
  -d '{"key":"demo-tenant|mock-cheap","capacity":1,"refill_rate":0.01}'

# First batch request uses the single token - routes to mock-cheap
curl -s -X POST http://127.0.0.1:8000/v1/complete \
  -H "X-API-Key: demo-key-123" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"x","messages":[{"role":"user","content":"Summarize the following document into bullet points"}]}' | python3 -m json.tool

# Second batch request - bucket is empty, router falls back
curl -s -X POST http://127.0.0.1:8000/v1/complete \
  -H "X-API-Key: demo-key-123" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"x","messages":[{"role":"user","content":"Summarize the following document into bullet points"}]}' | python3 -m json.tool
# expect: "fallback_used": true, "provider" is NOT "mock-cheap"

# Reset
curl -s -X POST http://127.0.0.1:8002/config \
  -H "Content-Type: application/json" \
  -d '{"key":"demo-tenant|mock-cheap","capacity":10,"refill_rate":1.0}'
```

**(d) Cost tracking — per-tenant rollup:**

```bash
curl -s http://127.0.0.1:8000/v1/cost/summary \
  -H "X-API-Key: demo-key-123" | python3 -m json.tool
```

**(e) Replication log — see the leader's decision stream:**

```bash
curl -s http://127.0.0.1:8002/log | python3 -m json.tool
curl -s http://127.0.0.1:8002/leader
```

### Step 6 — Shut down

In Terminal 1, hit **Ctrl-C**. All eight subprocesses will be signalled
and terminate cleanly.

### Troubleshooting

| Symptom                                              | Fix                                                                                       |
|------------------------------------------------------|-------------------------------------------------------------------------------------------|
| `ModuleNotFoundError: intelliroute`                  | Make sure you prefixed the command with `PYTHONPATH=.` and are in the `intelliroute/` root |
| `Address already in use` when starting               | Something else is on 8000-8004 or 9001-9003. Change the ports via env vars (see Configuration) |
| `httpx.ConnectError` from the demo                   | Terminal 1's stack didn't come up yet; wait for the "running" line                         |
| Integration tests hang                               | A previous run left subprocesses behind: `pkill -f "uvicorn intelliroute"`                |
| Unit tests pass but integration fails                | Almost always a port collision. Check `lsof -iTCP:8000-8004,9001-9003`                     |

---

## Project layout

```
intelliroute/
├── README.md                       <-- you are here
├── REPORT.docx                     <-- formal project report
├── pyproject.toml                  <-- package metadata, pytest config
├── intelliroute/                   <-- source code (the Python package)
│   ├── common/
│   │   ├── models.py               <-- Pydantic over-the-wire types
│   │   ├── config.py               <-- Settings / env-var parsing
│   │   └── logging.py              <-- JSON structured logger
│   ├── gateway/main.py             <-- Public-facing API (port 8000)
│   ├── router/
│   │   ├── main.py                 <-- Router service (port 8001)
│   │   ├── intent.py               <-- Intent classifier
│   │   ├── policy.py               <-- Multi-objective routing policy
│   │   └── registry.py             <-- In-memory service registry
│   ├── rate_limiter/
│   │   ├── main.py                 <-- Rate limiter service (port 8002)
│   │   └── token_bucket.py         <-- Token bucket + replication log
│   ├── cost_tracker/
│   │   ├── main.py                 <-- Cost tracker service (port 8003)
│   │   └── accounting.py           <-- Per-tenant rollups + budget alerts
│   ├── health_monitor/
│   │   ├── main.py                 <-- Health monitor service (port 8004)
│   │   └── circuit_breaker.py      <-- 3-state circuit breaker
│   └── mock_provider/main.py       <-- Simulated upstream LLM (ports 9001-9003)
├── tests/
│   ├── test_token_bucket.py        <-- 7 unit tests
│   ├── test_circuit_breaker.py     <-- 6 unit tests
│   ├── test_intent.py              <-- 7 unit tests
│   ├── test_policy.py              <-- 7 unit tests
│   ├── test_registry.py            <-- 4 unit tests
│   ├── test_accounting.py          <-- 4 unit tests
│   └── test_integration.py         <-- 7 end-to-end tests (spawn real services)
└── scripts/
    ├── start_stack.py              <-- Launch the full system
    └── demo.py                     <-- Demo client
```

## API surface

### Gateway (`:8000`) — the only service clients should talk to

| Method | Path                  | Description                                  |
|--------|-----------------------|----------------------------------------------|
| POST   | `/v1/complete`        | Submit an LLM completion (X-API-Key header)  |
| GET    | `/v1/cost/summary`    | Per-tenant cost rollup                       |
| GET    | `/v1/system/health`   | Aggregate provider health snapshot           |
| GET    | `/health`             | Liveness                                     |

### Router (`:8001`)

| Method | Path                  | Description                                                 |
|--------|-----------------------|-------------------------------------------------------------|
| POST   | `/complete`           | Internal: full routing + fallback execution                 |
| POST   | `/decide`             | Internal: returns the routing decision without executing it |
| GET    | `/providers`          | List registered providers                                   |
| POST   | `/providers`          | Register a provider                                         |
| DELETE | `/providers/{name}`   | Deregister a provider                                       |

### Rate Limiter (`:8002`)

| Method | Path        | Description                                                    |
|--------|-------------|----------------------------------------------------------------|
| POST   | `/check`    | Try to consume tokens for `(tenant, provider)`                |
| POST   | `/config`   | Update bucket capacity/refill for a key                        |
| GET    | `/leader`   | Current leader replica id                                      |
| GET    | `/log`      | Replication log (for follower replay)                          |

### Cost Tracker (`:8003`)

| Method | Path                   | Description                                  |
|--------|------------------------|----------------------------------------------|
| POST   | `/events`              | Publish a cost event                         |
| GET    | `/summary/{tenant_id}` | Per-tenant rollup                            |
| POST   | `/budget`              | Set a budget for a tenant                    |
| GET    | `/alerts`              | List budget alerts                           |

### Health Monitor (`:8004`)

| Method | Path                  | Description                                  |
|--------|-----------------------|----------------------------------------------|
| POST   | `/register`           | Register a provider URL for health polling   |
| POST   | `/report/{provider}`  | Report success/failure to update breaker     |
| GET    | `/snapshot`           | All breaker states                           |

### Mock Providers (`:9001-9003`)

| Method | Path                   | Description                                  |
|--------|------------------------|----------------------------------------------|
| POST   | `/v1/chat`             | Simulated LLM completion                     |
| POST   | `/admin/force_fail`    | Test hook to flip the provider into failing mode |
| GET    | `/health`              | Liveness                                     |

## Configuration

All services read configuration from environment variables. Defaults are
defined in `intelliroute/common/config.py`. The most important variables:

| Variable                              | Default          | Purpose                          |
|---------------------------------------|------------------|----------------------------------|
| `INTELLIROUTE_HOST`                   | `127.0.0.1`      | Bind host for every service      |
| `INTELLIROUTE_GATEWAY_PORT`           | `8000`           |                                  |
| `INTELLIROUTE_ROUTER_PORT`            | `8001`           |                                  |
| `INTELLIROUTE_RATE_LIMITER_PORT`      | `8002`           |                                  |
| `INTELLIROUTE_COST_TRACKER_PORT`      | `8003`           |                                  |
| `INTELLIROUTE_HEALTH_MONITOR_PORT`    | `8004`           |                                  |
| `INTELLIROUTE_MOCK_FAST_PORT`         | `9001`           | mock-fast provider               |
| `INTELLIROUTE_MOCK_SMART_PORT`        | `9002`           | mock-smart provider              |
| `INTELLIROUTE_MOCK_CHEAP_PORT`        | `9003`           | mock-cheap provider              |
| `INTELLIROUTE_DEMO_KEY`               | `demo-key-123`   | Demo API key for `demo-tenant`   |

---

## Team contributions

Work on IntelliRoute was divided across four subsystem owners. Each
teammate is the primary author of their modules and their corresponding
tests; everyone reviewed each other's pull requests and jointly wrote
the final report.

### Anukrithi Myadala — Gateway, shared infrastructure, and cross-cutting concerns

- Designed the shared Pydantic data model in `intelliroute/common/models.py`
  so every service speaks the same wire format.
- Implemented `intelliroute/common/config.py` (environment-variable
  driven settings, port allocation, service URLs).
- Built the structured JSON logger in `intelliroute/common/logging.py`
  used by every service.
- Wrote the API Gateway (`intelliroute/gateway/main.py`): API-key
  authentication, tenant identity rewriting, `X-Request-Id` trace
  propagation, cost-summary and system-health passthroughs.
- Wrote `scripts/start_stack.py` and `scripts/demo.py` and authored
  the README "Quick start for teammates" walkthrough.

### Larry Nguyen — Router service, routing policy, and intent classification

- Implemented the intent classifier in `intelliroute/router/intent.py`
  (hint-first, then code/batch/reasoning/interactive heuristics).
- Built the multi-objective routing policy in
  `intelliroute/router/policy.py`: normalised sub-scores for latency,
  cost, capability, and success rate, and intent-specific weight
  vectors.
- Implemented the provider registry in
  `intelliroute/router/registry.py` (the in-memory analogue of
  Consul/etcd for service discovery).
- Wrote the router service itself (`intelliroute/router/main.py`):
  bootstrap of the three mock providers, fallback loop that skips
  rate-limited or unhealthy providers, async reporting to the health
  monitor, and async cost-event publishing.
- Authored `tests/test_intent.py`, `tests/test_policy.py`, and
  `tests/test_registry.py` (18 unit tests).

### James Pham — Rate limiter, health monitor, leader/replication, circuit breakers

- Implemented the core `TokenBucket` and `RateLimiterStore`
  (`intelliroute/rate_limiter/token_bucket.py`), including the
  authoritative leader id and replication log used by follower
  replicas.
- Wrote the rate limiter service (`intelliroute/rate_limiter/main.py`):
  `/check`, `/config`, `/leader`, `/log` endpoints.
- Implemented the three-state circuit breaker in
  `intelliroute/health_monitor/circuit_breaker.py`
  (closed → open → half_open transitions with sliding error window).
- Wrote the health monitor service
  (`intelliroute/health_monitor/main.py`): `/report`, `/snapshot`,
  periodic background liveness polling of registered providers.
- Authored `tests/test_token_bucket.py` and
  `tests/test_circuit_breaker.py` (13 unit tests).

### Surbhi Singh — Cost tracker, mock providers, observability, and integration testing

- Implemented the `CostAccountant` in
  `intelliroute/cost_tracker/accounting.py` (per-tenant rollups,
  per-provider breakdown, budget-exceeded alerting).
- Wrote the cost tracker service (`intelliroute/cost_tracker/main.py`):
  `/events`, `/summary/{tenant_id}`, `/budget`, `/alerts`.
- Built the parametrised mock LLM provider
  (`intelliroute/mock_provider/main.py`) with configurable latency,
  jitter, failure rate, and cost per 1K tokens, plus the
  `/admin/force_fail` test hook.
- Authored `tests/test_accounting.py` (4 unit tests).
- Authored `tests/test_integration.py` — the 7 end-to-end tests that
  spawn the full 8-process stack on ephemeral ports and verify
  routing, failover, rate limiting, cost tracking, and authentication
  over real HTTP.
- Led the failure-mode demo scripts documented in "Quick start → Step 5".

### Joint contributions (all four teammates)

- Architecture design and weekly integration reviews.
- The project report (`REPORT.docx`) and the list of bugs caught during
  development (see the "Bug log" section of the report).
- Final demo preparation and presentation.

---

## What is intentionally simplified

For the scope of a one-semester project we deliberately stop short of:

- **A real Raft implementation.** The rate limiter exposes a leader id
  and replication log, but only one replica runs in the demo. Adding a
  follower that tails the log is straightforward and is called out as
  the natural extension.
- **A persistent store.** All state is in-memory; restarting a service
  resets it. Swapping in Redis or Postgres is a single-file change.
- **A learned intent classifier.** The classifier is a deterministic
  set of heuristics so that it can be unit tested without an external
  model.
- **Real LLM providers.** The mock providers simulate latency, cost,
  and failure modes parameterised by env vars; pointing the registry
  at a real provider is just a `ProviderInfo.url` change.

These scoping decisions keep the focus of the project where the course
expects it: on distributed-systems design and engineering tradeoffs.
