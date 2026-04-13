# Changelog

All notable changes to IntelliRoute are documented in this file.

---

## [0.2.0] - 2026-04-11

### Summary

Implements the three remaining spec gaps from the original project design: **feedback loop / continuous learning**,
**backpressure & queueing**, and **distributed coordination via leader
election**.  Test count grows from 42 to 83 (all passing).

---

### Added

#### Feedback Loop & Continuous Learning (`intelliroute/router/feedback.py`)
- `FeedbackCollector` class — accumulates per-provider performance metrics
  using an exponential moving average (EMA) with configurable smoothing
  factor (`alpha`).
- Tracks four live signals per provider:
  - **latency EMA** — observed completion latency in milliseconds.
  - **success rate EMA** — fraction of calls that succeeded.
  - **token efficiency EMA** — completion-to-prompt token ratio.
  - **anomaly score** — hallucination proxy based on response-to-prompt
    character-length ratio; flags responses that are abnormally short
    (< 10 % of prompt) or long (> 10x prompt).
- `CompletionOutcome` dataclass for recording individual observations.
- Thread-safe via `threading.Lock` (matches project convention).
- Injectable clock for deterministic unit testing.

#### Adaptive Routing Policy (`intelliroute/router/policy.py`)
- `RoutingPolicy` now accepts an optional `FeedbackCollector`.
- When feedback is available for a provider:
  - **Latency estimate** uses the feedback EMA instead of static config
    or health-monitor averages.
  - **Success score** uses the feedback EMA instead of the circuit-breaker
    error rate.
  - An **anomaly penalty** (`-0.1 * anomaly_score`) is applied to the
    composite score, demoting providers that produce suspect outputs.
- When feedback is `None` or a provider has not been observed yet, the
  policy falls back to the previous static behaviour (zero regression).

#### Router Feedback Integration (`intelliroute/router/main.py`)
- Module-level `FeedbackCollector` instance wired into the `RoutingPolicy`.
- Every successful completion records a `CompletionOutcome` (provider,
  latency, tokens, character counts).
- Every failed provider call records a failure outcome.
- New endpoint: **`GET /feedback`** — returns the current EMA metrics for
  every observed provider (JSON).

#### Priority Request Queue (`intelliroute/router/queue.py`)
- `RequestQueue` class — async-aware priority queue with three levels:
  - **HIGH** (interactive, code) — bypass the queue entirely.
  - **MEDIUM** (reasoning) — enqueued, processed by background workers.
  - **LOW** (batch) — enqueued, subject to load shedding.
- `QueueConfig` dataclass with tunable knobs:
  - `max_depth` (100) — absolute cap across all priorities.
  - `shed_threshold` (80) — start shedding LOW when total exceeds this.
  - `max_low_priority` (50) — cap on LOW items specifically.
  - `timeout_ms` (30 000) — max wait time before 504.
- `QueueStats` for observability (depth, per-priority counts, shed count,
  timeout count).
- `Priority` IntEnum and `QueuedRequest` dataclass for type-safe ordering.

#### Router Queue Integration (`intelliroute/router/main.py`)
- Refactored completion logic into `_execute_completion()` (pure async
  function, no FastAPI decorator).
- New `complete()` endpoint flow:
  - HIGH priority → immediate execution (bypasses queue).
  - MEDIUM / LOW → `try_enqueue()` → background worker picks it up.
  - Rejected requests get **HTTP 429** with `load_shed: <reason>`.
  - Queue timeouts get **HTTP 504** with `queue_timeout`.
- 4 background `_queue_worker` tasks started at service startup.
- New endpoint: **`GET /queue/stats`** — returns live queue statistics.

#### Bully Leader Election (`intelliroute/rate_limiter/election.py`)
- `LeaderElection` class — pure-logic Bully algorithm state machine.
- Three states: `FOLLOWER`, `CANDIDATE`, `LEADER`.
- Key operations:
  - `start_election()` — challenges all higher-ID peers; if none exist,
    declares self leader immediately.
  - `receive_challenge(from_id)` — responds whether I outrank the caller.
  - `receive_victory(leader_id)` — accepts a new leader, transitions to
    FOLLOWER.
  - `declare_victory()` — transitions to LEADER.
  - `receive_heartbeat(from_leader)` — resets the heartbeat timer.
  - `check_leader_timeout()` — returns True when the leader's heartbeat
    has expired (triggers re-election).
- `ElectionConfig` dataclass (election timeout, heartbeat interval,
  heartbeat timeout).
- `Peer` dataclass (replica ID + base URL).
- Thread-safe via `threading.Lock`; injectable clock for testing.

#### Rate Limiter Multi-Replica Support (`intelliroute/rate_limiter/`)
- `RateLimiterStore.set_leader(leader_id)` — update leader identity after
  an election.
- `RateLimiterStore.replay_log_entry(ts, key, amount, allowed)` — apply a
  leader's replication log entry on a follower without re-evaluating the
  bucket.
- `RateLimiterStore.log_length()` — return current log size (used by
  follower sync offset tracking).
- Rate limiter service (`main.py`) fully rewritten for multi-replica:
  - Reads `RATE_LIMITER_REPLICA_ID` and `RATE_LIMITER_PEERS` env vars.
  - `POST /check` — leader processes locally; followers forward to leader
    via HTTP (fail-open on leader unreachable).
  - Election endpoints: `POST /election/challenge`,
    `POST /election/victory`, `POST /election/heartbeat`,
    `GET /election/status`.
  - `GET /log/since/{offset}` — incremental replication log for followers.
  - Background tasks: `_heartbeat_loop` (leader pings followers every 1 s),
    `_leader_watchdog` (follower detects timeout → triggers election),
    `_log_sync_loop` (follower pulls and replays leader's log every 2 s).
  - Initial election runs automatically at startup.

#### Updated Stack Launcher (`scripts/start_stack.py`)
- Launches **3 rate limiter replicas** (ports 8002, 8012, 8022) instead of
  one, each with distinct `RATE_LIMITER_REPLICA_ID` and
  `RATE_LIMITER_PEERS` environment variables.
- Total services: 10 (was 8).

#### Updated Demo Client (`scripts/demo.py`)
- **Demo 1**: Intent-aware routing (unchanged behaviour, cleaner output).
- **Demo 2**: Feedback loop — queries `GET /feedback` and prints per-provider
  EMA metrics (latency, success rate, efficiency, anomaly score).
- **Demo 3**: Backpressure — sends 10 concurrent batch requests, reports
  how many succeeded vs. were shed (429), then shows an interactive request
  bypassing the queue.
- **Demo 4**: Leader election — queries `/election/status` on all 3 rate
  limiter replicas and prints which one is the elected leader.

#### New Tests

| File | Tests | What it covers |
|---|---|---|
| `tests/test_feedback.py` | 10 | EMA init, convergence, success rate, anomaly detection (short/long/normal), isolation, snapshots, token efficiency |
| `tests/test_queue.py` | 12 | Priority mapping (4 intents), enqueue acceptance, shed threshold, max depth, low-priority cap, dequeue ordering, stats, shed counter |
| `tests/test_election.py` | 11 | Follower start, highest-ID wins, challenge/response, victory acceptance, declare victory, heartbeat reset, leader timeout, leader self-check, higher-peer listing |
| `tests/test_policy.py` | +3 | Feedback latency override, anomaly penalty, fallback to static |
| `tests/test_token_bucket.py` | +2 | `set_leader`, `replay_log_entry` |
| `tests/test_integration.py` | +3 | Feedback endpoint populated, queue stats shape, election status shows leader |

**Total test count: 83** (was 42).

---

### Changed

- `RoutingPolicy.__init__` signature now accepts an optional `feedback`
  parameter.  Existing callers passing no feedback see identical behaviour.
- `RoutingPolicy.rank()` scoring logic prefers live feedback metrics over
  static config / health-monitor values when available.
- `router/main.py` — `complete()` refactored into queue-aware dispatcher +
  `_execute_completion()` core logic.
- `rate_limiter/main.py` — complete rewrite to support leader election and
  follower forwarding (backwards-compatible: single-replica with no peers
  behaves exactly as before).
- `scripts/start_stack.py` — rate limiter section expanded to 3 replicas.
- `scripts/demo.py` — expanded with 3 new demo sections.
- `pyproject.toml` — version bumped from `0.1.0` to `0.2.0`.

---

### Spec Coverage (before → after)

| Component | v0.1.0 | v0.2.0 |
|---|---|---|
| Intent-Aware Routing | Done | Done |
| Global Rate Limit Coordination | Done | Done |
| Adaptive Fallback & Degradation | Done | Done |
| Cost-Aware Scheduling | Done | Done |
| Observability & Feedback Loop | Partial | **Done** |
| Service Discovery | Done | Done |
| Distributed Coordination | Partial | **Done** |
| Consistency Models | Partial | **Done** |
| Fault Tolerance | Done | Done |
| Backpressure & Queueing | Not done | **Done** |
| Multi-Objective Optimization | Done | Done |

---

## [0.1.0] - 2026-04-11

Initial release.  Core distributed LLM gateway with intent-aware routing,
token-bucket rate limiting, circuit breakers, cost tracking, and mock
providers.  42 tests (35 unit + 7 integration).
