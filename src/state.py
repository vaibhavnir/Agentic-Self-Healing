"""
Shared state definition for the self-healing LangGraph workflow.
All nodes read from and write to this TypedDict.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class HealingState(TypedDict):
    """Immutable snapshot of the pipeline's healing context passed between nodes."""

    # Input: raw error signal from the ETL pipeline
    error_signal: dict[str, Any]

    # Diagnosis produced by the 'diagnose' node
    diagnosed_issue: str

    # Canonical error type key (maps to strategy config)
    error_type: str

    # List of strategy names evaluated by 'plan_recovery'
    available_actions: list[str]

    # Strategy chosen for execution
    selected_action: str

    # Parameters for the chosen strategy (merged from config + runtime)
    action_parameters: dict[str, Any]

    # Outcome produced by the 'execute' node
    execution_result: dict[str, Any]

    # Confidence that the chosen action will succeed (0.0 – 1.0)
    confidence_score: float

    # Number of recovery attempts for this event
    attempt_count: int

    # True once the pipeline is back in a healthy state
    is_recovered: bool

    # Reasoning trace for auditability
    reasoning_trace: list[str]

    # Databricks job integration (optional – populated when error originates from a Databricks job)
    databricks_job_id: NotRequired[str]
    databricks_run_id: NotRequired[str]
