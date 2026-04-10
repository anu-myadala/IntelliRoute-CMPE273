"""Unit tests for the token-bucket rate limiter."""
from __future__ import annotations

from intelliroute.rate_limiter.token_bucket import (
    BucketConfig,
    RateLimiterStore,
    TokenBucket,
)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_bucket_starts_full_on_first_use():
    bucket = TokenBucket(capacity=10, refill_rate=1)
    allowed, remaining, retry = bucket.try_consume(1, now=100.0)
    assert allowed is True
    assert remaining == 9
    assert retry == 0


def test_bucket_denies_when_empty_and_reports_retry_after():
    bucket = TokenBucket(capacity=2, refill_rate=1)  # 1 token/sec
    assert bucket.try_consume(1, now=0.0)[0] is True  # 1 remaining
    assert bucket.try_consume(1, now=0.0)[0] is True  # 0 remaining
    allowed, remaining, retry = bucket.try_consume(1, now=0.0)
    assert allowed is False
    assert remaining == 0
    # 1 token needed at 1 token/sec -> ~1000 ms
    assert 1000 <= retry <= 1100


def test_bucket_refills_linearly_over_time():
    bucket = TokenBucket(capacity=10, refill_rate=5)  # 5/sec
    # Drain it
    assert bucket.try_consume(10, now=0.0)[0] is True
    # 1 second later we should have 5 tokens.
    allowed, remaining, _ = bucket.try_consume(5, now=1.0)
    assert allowed is True
    assert abs(remaining - 0) < 1e-9


def test_bucket_capacity_cap():
    bucket = TokenBucket(capacity=10, refill_rate=5)
    assert bucket.try_consume(10, now=0.0)[0] is True
    # Wait way past the fill time; should cap at 10.
    allowed, remaining, _ = bucket.try_consume(0, now=100.0)
    assert allowed is True
    assert remaining == 10


def test_store_isolates_keys():
    clock = FakeClock()
    store = RateLimiterStore(
        default_config=BucketConfig(capacity=2, refill_rate=1),
        clock=clock,
    )
    # Tenant A drains its bucket
    assert store.try_consume("tenantA|p1")[0] is True
    assert store.try_consume("tenantA|p1")[0] is True
    assert store.try_consume("tenantA|p1")[0] is False
    # Tenant B is unaffected
    assert store.try_consume("tenantB|p1")[0] is True


def test_store_custom_config_per_key():
    clock = FakeClock()
    store = RateLimiterStore(
        default_config=BucketConfig(capacity=1, refill_rate=1),
        configs={"vip|p1": BucketConfig(capacity=100, refill_rate=50)},
        clock=clock,
    )
    # VIP can burst to 100
    for _ in range(100):
        assert store.try_consume("vip|p1")[0] is True
    assert store.try_consume("vip|p1")[0] is False


def test_store_logs_decisions_for_replication():
    clock = FakeClock()
    store = RateLimiterStore(
        default_config=BucketConfig(capacity=2, refill_rate=1),
        clock=clock,
    )
    store.try_consume("t|p")
    store.try_consume("t|p")
    store.try_consume("t|p")  # denied
    log = store.replication_log()
    assert len(log) == 3
    assert [entry[3] for entry in log] == [True, True, False]
