"""Shared Pydantic data models used across IntelliRoute services.

These are the canonical over-the-wire types. Every service imports from here
so that the gateway, router, rate limiter, cost tracker, and health monitor
all speak the same language.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Intent(str, Enum):
    """High-level classification of a prompt.

    The routing policy uses the intent to pick an appropriate provider/model.
    """

    INTERACTIVE = "interactive"  # low-latency, short responses (chat UI)
    REASONING = "reasoning"      # high-accuracy, longer chain-of-thought
    BATCH = "batch"              # offline/bulk, cost-sensitive
    CODE = "code"                # code generation / completion


class ChatMessage(BaseModel):
    role: str = Field(..., description="'system', 'user', or 'assistant'")
    content: str


class CompletionRequest(BaseModel):
    """A unified request format that mirrors the common shape of LLM APIs."""

    tenant_id: str = Field(..., description="Identifier of the calling tenant/team")
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    # Optional explicit hint from the caller
    intent_hint: Optional[Intent] = None
    # Caller can set a hard latency budget in ms; router may use it to pick faster models
    latency_budget_ms: Optional[int] = Field(None, ge=100, le=30000)

    @field_validator("tenant_id")
    @classmethod
    def tenant_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("tenant_id must not be blank")
        return v


class CompletionResponse(BaseModel):
    request_id: str
    provider: str
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    # Set when the router had to fall back from the preferred provider
    fallback_used: bool = False
    degraded: bool = False


class ProviderInfo(BaseModel):
    """A registered LLM backend."""

    name: str                    # e.g. "mock-openai"
    url: str                     # base URL of the provider service
    model: str                   # model identifier
    # Capability scores in [0, 1]. Higher = better at that intent.
    capability: dict[str, float] = Field(default_factory=dict)
    # Cost in USD per 1K tokens (input + output averaged for simplicity)
    cost_per_1k_tokens: float = 0.0
    # Expected latency for a short completion, milliseconds
    typical_latency_ms: float = 500.0
    # Max concurrent in-flight requests recommended
    max_concurrency: int = 32


class ProviderHealth(BaseModel):
    name: str
    healthy: bool
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    circuit_state: str = "closed"  # closed / open / half_open
    consecutive_failures: int = 0
    last_checked_unix: float = 0.0


class RateLimitCheck(BaseModel):
    tenant_id: str
    provider: str
    tokens_requested: int = 1


class RateLimitResult(BaseModel):
    allowed: bool
    remaining: float
    retry_after_ms: int = 0
    leader_replica: Optional[str] = None


class CostEvent(BaseModel):
    request_id: str
    tenant_id: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    unix_ts: float


class CostSummary(BaseModel):
    tenant_id: str
    total_requests: int
    total_tokens: int
    total_cost_usd: float
    by_provider: dict[str, float]
