"""
Pydantic request / response models for the self-healing API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class HealRequest(BaseModel):
    """Single failure to heal."""

    error_signal: dict[str, Any] = Field(
        ...,
        description="Raw error signal dict from the ETL pipeline.",
        examples=[
            {
                "type": "timeout",
                "source": "postgres_loader",
                "message": "Query timed out after 30s",
                "last_checkpoint": "cp_001",
            }
        ],
    )


class BatchHealRequest(BaseModel):
    """Multiple failures to heal in parallel."""

    failures: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of raw error signal dicts (processed concurrently).",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealResponse(BaseModel):
    """Outcome of a single healing attempt."""

    is_recovered: bool = Field(description="True when the pipeline recovered successfully.")
    error_type: str = Field(description="Canonical error type that was diagnosed.")
    diagnosed_issue: str = Field(description="Human-readable diagnosis.")
    selected_action: str = Field(description="Recovery strategy that was chosen.")
    confidence_score: float = Field(description="Confidence score for the selected action.")
    attempt_count: int = Field(description="Total number of recovery attempts made.")
    execution_result: dict[str, Any] = Field(description="Raw result from the strategy execution.")
    reasoning_trace: list[str] = Field(description="Step-by-step reasoning log for auditability.")
    databricks_run_id: str | None = Field(
        default=None,
        description="Databricks run ID when a job was re-triggered.",
    )


class BatchHealResponse(BaseModel):
    """Outcomes for all failures in a batch request."""

    results: list[HealResponse]
    total: int = Field(description="Total number of failures processed.")
    recovered: int = Field(description="Number that recovered successfully.")
    escalated: int = Field(description="Number that required human escalation.")


class OutcomeSummaryRow(BaseModel):
    """Single row in the adaptive healer outcome summary."""

    error_type: str
    action: str
    successes: int
    failures: int
    total: int
    success_rate: float


class HealthResponse(BaseModel):
    status: str = "ok"
    workflow_ready: bool = False
