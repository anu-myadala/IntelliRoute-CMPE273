"""Unit tests for the provider registry (service discovery)."""
from __future__ import annotations

from intelliroute.common.models import ProviderInfo
from intelliroute.router.registry import ProviderRegistry


def _p(name: str) -> ProviderInfo:
    return ProviderInfo(name=name, url=f"http://{name}", model=f"{name}-m")


def test_register_and_get():
    reg = ProviderRegistry()
    reg.register(_p("a"))
    assert reg.get("a") is not None
    assert reg.get("missing") is None


def test_deregister():
    reg = ProviderRegistry()
    reg.register(_p("a"))
    reg.deregister("a")
    assert reg.get("a") is None


def test_bulk_register_and_all():
    reg = ProviderRegistry()
    reg.bulk_register([_p("a"), _p("b"), _p("c")])
    names = sorted(p.name for p in reg.all())
    assert names == ["a", "b", "c"]


def test_re_register_overwrites():
    reg = ProviderRegistry()
    reg.register(_p("a"))
    updated = ProviderInfo(name="a", url="http://new", model="a-m2")
    reg.register(updated)
    assert reg.get("a").url == "http://new"
