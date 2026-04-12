"""Unit tests for the Bully leader election algorithm."""
from __future__ import annotations

from intelliroute.rate_limiter.election import (
    ElectionConfig,
    ElectionState,
    LeaderElection,
    Peer,
)


class _Clock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_election(
    my_id: str = "rl-1",
    peer_ids: list[str] | None = None,
    config: ElectionConfig | None = None,
) -> tuple[LeaderElection, _Clock]:
    if peer_ids is None:
        peer_ids = ["rl-0", "rl-2"]
    clock = _Clock()
    peers = [Peer(replica_id=pid, url=f"http://{pid}") for pid in peer_ids]
    cfg = config or ElectionConfig(
        election_timeout_s=2.0,
        heartbeat_interval_s=1.0,
        heartbeat_timeout_s=3.0,
    )
    return LeaderElection(my_id=my_id, peers=peers, config=cfg, clock=clock), clock


def test_starts_as_follower():
    e, _ = _make_election()
    assert e.state == ElectionState.FOLLOWER
    assert e.current_leader is None


def test_highest_id_becomes_leader_immediately():
    """If no peer has a higher ID, start_election declares victory."""
    e, _ = _make_election(my_id="rl-9", peer_ids=["rl-0", "rl-1"])
    higher = e.start_election()
    assert higher == []
    assert e.state == ElectionState.LEADER
    assert e.current_leader == "rl-9"
    assert e.is_leader is True


def test_lower_id_challenges_higher():
    """A lower-ID replica should list higher peers to challenge."""
    e, _ = _make_election(my_id="rl-0", peer_ids=["rl-1", "rl-2"])
    higher = e.start_election()
    assert set(higher) == {"rl-1", "rl-2"}
    assert e.state == ElectionState.CANDIDATE


def test_receive_challenge_from_lower_returns_true():
    """If a lower-ID replica challenges me, I respond True (I'm alive)."""
    e, _ = _make_election(my_id="rl-2")
    assert e.receive_challenge("rl-0") is True


def test_receive_challenge_from_higher_returns_false():
    """If a higher-ID replica challenges me, I yield."""
    e, _ = _make_election(my_id="rl-0")
    assert e.receive_challenge("rl-2") is False


def test_victory_announcement_sets_follower():
    e, _ = _make_election(my_id="rl-0")
    e.start_election()
    e.receive_victory("rl-2")
    assert e.state == ElectionState.FOLLOWER
    assert e.current_leader == "rl-2"


def test_declare_victory():
    e, _ = _make_election(my_id="rl-1")
    e.start_election()
    e.declare_victory()
    assert e.state == ElectionState.LEADER
    assert e.current_leader == "rl-1"


def test_heartbeat_resets_timeout():
    e, clock = _make_election(my_id="rl-0")
    e.receive_victory("rl-2")  # become follower
    clock.advance(2.0)
    e.receive_heartbeat("rl-2")
    # Should not timeout since heartbeat was just received
    assert e.check_leader_timeout() is False


def test_leader_timeout_detected():
    e, clock = _make_election(my_id="rl-0")
    e.receive_victory("rl-2")  # become follower
    clock.advance(4.0)  # exceeds heartbeat_timeout_s=3.0
    assert e.check_leader_timeout() is True


def test_leader_does_not_timeout():
    """Leaders should never report a timeout on themselves."""
    e, clock = _make_election(my_id="rl-9", peer_ids=["rl-0"])
    e.start_election()  # becomes leader
    clock.advance(100.0)
    assert e.check_leader_timeout() is False


def test_higher_peers():
    e, _ = _make_election(my_id="rl-1", peer_ids=["rl-0", "rl-2", "rl-3"])
    higher = e.higher_peers()
    ids = [p.replica_id for p in higher]
    assert set(ids) == {"rl-2", "rl-3"}
