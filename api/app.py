"""
FastAPI application for the Agentic Self-Healing ETL pipeline.

Routes
------
GET  /health          – liveness probe
POST /heal            – heal a single failure
POST /heal/batch      – heal multiple failures concurrently (thread-safe)
GET  /outcomes        – adaptive healer learning summary

Run
---
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status

from src.state import HealingState

from .dependencies import app_state
from .models import (
    BatchHealRequest,
    BatchHealResponse,
    HealRequest,
    HealResponse,
    HealthResponse,
    OutcomeSummaryRow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan – initialise shared state once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):  # noqa: ANN001
    """Build the workflow and healer on startup; nothing to clean up on shutdown."""
    logger.info("Self-Healing API: initialising workflow…")
    app_state.initialise()
    logger.info("Self-Healing API: workflow ready")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic Self-Healing ETL API",
    description=(
        "LangGraph-powered self-healing for Databricks ETL pipelines. "
        "Supports parallel failure processing with full thread safety."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_initial_state(error_signal: dict[str, Any]) -> HealingState:
    return {
        "error_signal": error_signal,
        "diagnosed_issue": "",
        "error_type": "",
        "available_actions": [],
        "selected_action": "",
        "action_parameters": {},
        "execution_result": {},
        "confidence_score": 0.0,
        "attempt_count": 0,
        "is_recovered": False,
        "reasoning_trace": [],
    }


def _state_to_response(state: HealingState) -> HealResponse:
    return HealResponse(
        is_recovered=state.get("is_recovered", False),
        error_type=state.get("error_type", ""),
        diagnosed_issue=state.get("diagnosed_issue", ""),
        selected_action=state.get("selected_action", ""),
        confidence_score=state.get("confidence_score", 0.0),
        attempt_count=state.get("attempt_count", 0),
        execution_result=state.get("execution_result", {}),
        reasoning_trace=state.get("reasoning_trace", []),
        databricks_run_id=state.get("execution_result", {}).get("metadata", {}).get("run_id"),
    )


async def _run_workflow(error_signal: dict[str, Any]) -> HealResponse:
    """Invoke the shared compiled workflow for a single error signal."""
    final_state: HealingState = await app_state.workflow.ainvoke(
        _make_initial_state(error_signal)
    )
    return _state_to_response(final_state)


def _check_ready() -> None:
    if not app_state.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Healing workflow not yet initialised",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["ops"],
)
async def health() -> HealthResponse:
    """Return service health and readiness status."""
    return HealthResponse(status="ok", workflow_ready=app_state.ready)


@app.post(
    "/heal",
    response_model=HealResponse,
    status_code=status.HTTP_200_OK,
    summary="Heal a single ETL failure",
    tags=["healing"],
)
async def heal_single(request: HealRequest) -> HealResponse:
    """
    Run the self-healing workflow for a single error signal.

    The workflow diagnoses the issue, selects the best recovery strategy
    (optionally using an LLM), executes it, and returns the full result
    with a reasoning trace.
    """
    _check_ready()
    logger.info("/heal: error_type=%s source=%s",
                request.error_signal.get("type"), request.error_signal.get("source"))
    return await _run_workflow(request.error_signal)


@app.post(
    "/heal/batch",
    response_model=BatchHealResponse,
    status_code=status.HTTP_200_OK,
    summary="Heal multiple ETL failures in parallel",
    tags=["healing"],
)
async def heal_batch(request: BatchHealRequest) -> BatchHealResponse:
    """
    Run the self-healing workflow for every failure in the request concurrently.

    All failures are dispatched simultaneously via ``asyncio.gather`` so total
    latency is bounded by the slowest individual failure, not their sum.
    The shared ``AdaptiveHealer`` is protected by a ``threading.Lock`` ensuring
    that concurrent outcome recordings remain consistent.
    """
    _check_ready()
    logger.info("/heal/batch: processing %d failures concurrently", len(request.failures))

    results: list[HealResponse] = await asyncio.gather(
        *[_run_workflow(signal) for signal in request.failures]
    )

    recovered = sum(1 for r in results if r.is_recovered)
    return BatchHealResponse(
        results=results,
        total=len(results),
        recovered=recovered,
        escalated=len(results) - recovered,
    )


@app.get(
    "/outcomes",
    response_model=list[OutcomeSummaryRow],
    summary="Adaptive healer learning summary",
    tags=["ops"],
)
async def outcomes() -> list[OutcomeSummaryRow]:
    """
    Return the adaptive healer's accumulated outcome history, sorted by
    success rate descending.  Use this to understand which strategies are
    working well for which error types.
    """
    _check_ready()
    rows = app_state.healer.summary()  # type: ignore[union-attr]
    return [OutcomeSummaryRow(**row) for row in rows]
