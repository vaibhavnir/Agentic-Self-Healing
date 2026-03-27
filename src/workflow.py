"""
LangGraph workflow definition for the self-healing ETL pipeline.

Graph topology
--------------

  [detect] → [diagnose] → [plan_recovery] → [execute] → [validate]
                                                              |
                               ┌──────────────────────────────┘
                               ↓
                     confidence ≥ threshold
                     AND attempt < max     →  [execute]  (retry loop)
                               ↓
                     !recovered AND attempts exhausted
                               →  [escalate] → END
                               ↓
                     recovered → END
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from langgraph.graph import END, StateGraph

from .adaptive_healer import AdaptiveHealer
from .nodes import (
    detect_issue,
    diagnose_issue,
    escalate_to_human,
    execute_recovery,
    plan_recovery,
    validate_recovery,
)
from .state import HealingState
from .strategies import RecoveryStrategy, build_strategy_registry

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 5
_CONFIDENCE_RETRY_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def _route_after_validate(
    state: HealingState,
    max_attempts: int,
    confidence_threshold: float,
) -> str:
    """
    Decide what to do after validation:
      - "done"     → pipeline recovered
      - "retry"    → confidence is high enough and we have attempts left
      - "escalate" → give up and alert humans
    """
    if state.get("is_recovered"):
        return "done"

    attempt = state.get("attempt_count", 0)
    confidence = state.get("confidence_score", 0.0)

    if attempt < max_attempts and confidence >= confidence_threshold:
        logger.info(
            "_route_after_validate: retrying (attempt=%d, confidence=%.2f)",
            attempt,
            confidence,
        )
        return "retry"

    return "escalate"


# ---------------------------------------------------------------------------
# Workflow factory
# ---------------------------------------------------------------------------

def create_self_healing_workflow(
    strategy_config: list[dict[str, Any]],
    adaptive_healer: AdaptiveHealer,
    max_attempts: int = _MAX_ATTEMPTS,
    confidence_threshold: float = _CONFIDENCE_RETRY_THRESHOLD,
) -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Parameters
    ----------
    strategy_config:
        Raw list of strategy dicts loaded from ``healing_strategies.yaml``.
    adaptive_healer:
        Shared AdaptiveHealer instance that persists learning across runs.
    max_attempts:
        Maximum number of recovery attempts before escalation.
    confidence_threshold:
        Minimum confidence score required to retry instead of escalating.

    Returns
    -------
    Compiled LangGraph runnable (supports ``ainvoke`` / ``invoke``).
    """
    strategy_registry: dict[str, RecoveryStrategy] = build_strategy_registry(
        strategy_config
    )
    known_types: list[str] = _collect_known_types(strategy_config)

    graph = StateGraph(HealingState)

    # ------------------------------------------------------------------
    # Register nodes (bind dependencies via functools.partial)
    # ------------------------------------------------------------------

    graph.add_node("detect", detect_issue)

    graph.add_node(
        "diagnose",
        functools.partial(diagnose_issue, known_types=known_types),
    )

    graph.add_node(
        "plan_recovery",
        functools.partial(
            plan_recovery,
            strategy_registry=strategy_registry,
            adaptive_healer=adaptive_healer,
        ),
    )

    graph.add_node(
        "execute",
        functools.partial(execute_recovery, strategy_registry=strategy_registry),
    )

    graph.add_node(
        "validate",
        functools.partial(
            validate_recovery,
            adaptive_healer=adaptive_healer,
            confidence_threshold=confidence_threshold,
        ),
    )

    graph.add_node("escalate", escalate_to_human)

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    graph.set_entry_point("detect")
    graph.add_edge("detect", "diagnose")
    graph.add_edge("diagnose", "plan_recovery")
    graph.add_edge("plan_recovery", "execute")
    graph.add_edge("execute", "validate")

    # Conditional routing after validation
    router = functools.partial(
        _route_after_validate,
        max_attempts=max_attempts,
        confidence_threshold=confidence_threshold,
    )

    graph.add_conditional_edges(
        "validate",
        router,
        {
            "done": END,
            "retry": "execute",      # loop back for another attempt
            "escalate": "escalate",
        },
    )

    graph.add_edge("escalate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_known_types(strategy_config: list[dict[str, Any]]) -> list[str]:
    """Flatten all applicable_errors across strategy configs into a deduplicated list."""
    seen: set[str] = set()
    result: list[str] = []
    for entry in strategy_config:
        for et in entry.get("applicable_errors", []):
            if et not in seen:
                seen.add(et)
                result.append(et)
    return result
