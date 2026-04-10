"""In-memory service registry for LLM providers.

This plays the role of Consul/etcd in the course's "service discovery"
topic. Providers register themselves, and the router queries by intent to
find candidates. Health information is applied on top of the registry by
the routing policy.
"""
from __future__ import annotations

import threading
from typing import Iterable, Optional

from ..common.models import ProviderInfo


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, ProviderInfo] = {}
        self._lock = threading.Lock()

    def register(self, info: ProviderInfo) -> None:
        with self._lock:
            self._providers[info.name] = info

    def deregister(self, name: str) -> None:
        with self._lock:
            self._providers.pop(name, None)

    def get(self, name: str) -> Optional[ProviderInfo]:
        with self._lock:
            return self._providers.get(name)

    def all(self) -> list[ProviderInfo]:
        with self._lock:
            return list(self._providers.values())

    def bulk_register(self, providers: Iterable[ProviderInfo]) -> None:
        with self._lock:
            for p in providers:
                self._providers[p.name] = p
