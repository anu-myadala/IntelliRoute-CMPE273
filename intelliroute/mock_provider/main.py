"""Mock LLM provider service.

Each instance of this service simulates a single upstream LLM backend
(e.g. OpenAI, Anthropic, a local Llama). The behaviour is parameterised
at startup by environment variables so we can spin up three flavours
with different latency/cost/failure profiles for the demo and tests.

Environment variables
---------------------
MOCK_NAME          -- provider identifier, e.g. "mock-fast"
MOCK_MODEL         -- model identifier, e.g. "fast-1"
MOCK_LATENCY_MS    -- average simulated latency, milliseconds
MOCK_LATENCY_JITTER_MS -- random +/- jitter added to latency
MOCK_FAILURE_RATE  -- probability in [0, 1] that a request returns 503
MOCK_COST_PER_1K   -- cost per 1K tokens (used in logs only)
"""
from __future__ import annotations

import asyncio
import os
import random
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..common.logging import get_logger, log_event


class MockChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 256


class MockChatResponse(BaseModel):
    id: str
    provider: str
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


NAME = os.environ.get("MOCK_NAME", "mock-provider")
MODEL = os.environ.get("MOCK_MODEL", "mock-1")
LATENCY_MS = _env_float("MOCK_LATENCY_MS", 100.0)
JITTER_MS = _env_float("MOCK_LATENCY_JITTER_MS", 20.0)
FAILURE_RATE = _env_float("MOCK_FAILURE_RATE", 0.0)
COST_PER_1K = _env_float("MOCK_COST_PER_1K", 0.001)

log = get_logger(NAME)
app = FastAPI(title=f"IntelliRoute MockProvider {NAME}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"force_fail": False}


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy" if not _state["force_fail"] else "degraded", "provider": NAME}


class ForceFailBody(BaseModel):
    fail: bool = True


@app.post("/admin/force_fail")
async def force_fail(body: ForceFailBody = ForceFailBody()) -> dict:
    """Test hook: flip the provider into failing mode."""
    _state["force_fail"] = body.fail
    return {"force_fail": body.fail}


@app.post("/v1/chat", response_model=MockChatResponse)
async def chat(req: MockChatRequest) -> MockChatResponse:
    # Simulate latency.
    latency = max(0.0, (LATENCY_MS + random.uniform(-JITTER_MS, JITTER_MS)) / 1000.0)
    await asyncio.sleep(latency)

    # Simulate failures.
    if _state["force_fail"] or random.random() < FAILURE_RATE:
        log_event(log, "simulated_failure", provider=NAME)
        raise HTTPException(status_code=503, detail=f"{NAME} temporarily unavailable")

    prompt_text = " ".join(m.get("content", "") for m in req.messages)
    prompt_tokens = max(1, len(prompt_text) // 4)
    completion_tokens = min(req.max_tokens, max(1, prompt_tokens // 2))
    content = (
        f"[{NAME}:{MODEL}] synthesized reply to: "
        f"{prompt_text[:80]!r} ({completion_tokens} tokens)"
    )

    resp = MockChatResponse(
        id=str(uuid.uuid4()),
        provider=NAME,
        model=MODEL,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    log_event(
        log,
        "completion_ok",
        provider=NAME,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=round(latency * 1000, 1),
    )
    return resp
