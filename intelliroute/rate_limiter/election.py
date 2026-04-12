"""Bully-algorithm leader election for rate-limiter replicas.

Pure-logic state machine with no I/O.  The HTTP communication layer
(challenging peers, broadcasting victory, heartbeats) lives in the
service layer (``main.py``), keeping this module fully unit-testable.

Each replica has a unique string ID.  The replica with the
lexicographically **highest** ID wins an election—matching the classic
Bully algorithm where the "biggest" process becomes coordinator.

Thread safety is provided by a single ``threading.Lock``, consistent
with ``CircuitBreaker``, ``RateLimiterStore``, and other stateful
classes in the project.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


Clock = Callable[[], float]


@dataclass
class Peer:
    """A known rate-limiter replica."""

    replica_id: str
    url: str  # base URL, e.g. "http://127.0.0.1:8012"


class ElectionState(str, Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class ElectionConfig:
    election_timeout_s: float = 2.0    # wait for higher-ID responses
    heartbeat_interval_s: float = 1.0  # leader pings followers
    heartbeat_timeout_s: float = 3.0   # follower detects leader failure


class LeaderElection:
    """Bully-algorithm leader election state machine.

    Parameters
    ----------
    my_id : str
        Unique identifier for this replica.
    peers : list[Peer]
        All other known replicas.
    config : ElectionConfig, optional
        Timing parameters.
    clock : callable, optional
        Injectable clock for deterministic testing.
    """

    def __init__(
        self,
        my_id: str,
        peers: list[Peer],
        config: Optional[ElectionConfig] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self._my_id = my_id
        self._peers = {p.replica_id: p for p in peers}
        self._config = config or ElectionConfig()
        self._clock: Clock = clock or time.monotonic
        self._state = ElectionState.FOLLOWER
        self._current_leader: Optional[str] = None
        self._last_heartbeat: float = self._clock()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def my_id(self) -> str:
        return self._my_id

    @property
    def state(self) -> ElectionState:
        with self._lock:
            return self._state

    @property
    def current_leader(self) -> Optional[str]:
        with self._lock:
            return self._current_leader

    @property
    def is_leader(self) -> bool:
        with self._lock:
            return self._state == ElectionState.LEADER

    @property
    def peers(self) -> list[Peer]:
        return list(self._peers.values())

    # ------------------------------------------------------------------
    # Election protocol
    # ------------------------------------------------------------------

    def higher_peers(self) -> list[Peer]:
        """Return peers whose ID is lexicographically greater than mine."""
        return [p for p in self._peers.values() if p.replica_id > self._my_id]

    def start_election(self) -> list[str]:
        """Begin an election.

        Returns the IDs of higher peers to challenge.  If there are
        none, the caller should invoke ``declare_victory`` immediately.
        """
        with self._lock:
            self._state = ElectionState.CANDIDATE
            higher = [
                pid for pid in self._peers if pid > self._my_id
            ]
            if not higher:
                # I am the highest; declare victory immediately.
                self._state = ElectionState.LEADER
                self._current_leader = self._my_id
            return higher

    def receive_challenge(self, from_id: str) -> bool:
        """Handle an election challenge from ``from_id``.

        Returns ``True`` if I am alive and have a higher ID (the
        challenger should back off).  Returns ``False`` if the
        challenger has a higher ID than me.
        """
        with self._lock:
            if self._my_id > from_id:
                # I outrank the challenger.  I should start my own election
                # (the caller in main.py handles that).
                return True
            return False

    def receive_victory(self, leader_id: str) -> None:
        """Accept a victory announcement from the new leader."""
        with self._lock:
            self._current_leader = leader_id
            if leader_id == self._my_id:
                self._state = ElectionState.LEADER
            else:
                self._state = ElectionState.FOLLOWER
                self._last_heartbeat = self._clock()

    def declare_victory(self) -> None:
        """No higher peer responded—I am the leader."""
        with self._lock:
            self._state = ElectionState.LEADER
            self._current_leader = self._my_id

    def receive_heartbeat(self, from_leader: str) -> None:
        """Update last-heartbeat timestamp from the current leader."""
        with self._lock:
            self._last_heartbeat = self._clock()
            # Accept the leader identity (handles late arrivals)
            if self._current_leader != from_leader:
                self._current_leader = from_leader
            if self._state != ElectionState.LEADER:
                self._state = ElectionState.FOLLOWER

    def check_leader_timeout(self) -> bool:
        """Return ``True`` if the leader's heartbeat has expired.

        Only meaningful for FOLLOWER replicas.  If the caller gets
        ``True`` it should trigger a new election.
        """
        with self._lock:
            if self._state != ElectionState.FOLLOWER:
                return False
            elapsed = self._clock() - self._last_heartbeat
            return elapsed > self._config.heartbeat_timeout_s
