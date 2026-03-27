"""
LangGraph node functions for the self-healing ETL workflow.

Each function takes a HealingState and returns a (possibly partial) dict
that LangGraph merges back into the state.
"""

from __future__ import annotations

import logging
from typing import Any

from .adaptive_healer import AdaptiveHealer
from .llm_interface import diagnose_with_llm, select_strategy_with_llm
from .state import HealingState
from .strategies import RecoveryStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node: detect
# ---------------------------------------------------------------------------

def detect_issue(state: HealingState) -> dict[str, Any]:
    """
    Validate and enrich the incoming error signal.
    Ensures required fields exist and normalises the signal.
    """
    signal = dict(state.get("error_signal", {}))

    # Normalise mandatory fields
    signal.setdefault("type", "unknown")
    signal.setdefault("message", "No message provided")
    signal.setdefault("source", "unknown")
    signal.setdefault("timestamp", None)

    trace = list(state.get("reasoning_trace", []))
    trace.append(
        f"[detect] Error signal received from '{signal['source']}': "
        f"type='{signal['type']}', message='{signal['message']}'"
    )

    logger.info("detect_issue: %s", trace[-1])

    return {
        "error_signal": signal,
        "attempt_count": state.get("attempt_count", 0),
        "is_recovered": False,
        "reasoning_trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: diagnose
# ---------------------------------------------------------------------------

def diagnose_issue(
    state: HealingState,
    known_types: list[str],
) -> dict[str, Any]:
    """
    Use the LLM (or config-based fallback) to produce a diagnosis and
    canonical error_type for the received error signal.
    """
    result = diagnose_with_llm(state["error_signal"], known_types)

    trace = list(state.get("reasoning_trace", []))
    trace.append(
        f"[diagnose] Issue: '{result['diagnosed_issue']}' "
        f"| error_type='{result['error_type']}' "
        f"| root_cause='{result.get('root_cause_explanation', '')}'"
    )
    logger.info("diagnose_issue: %s", trace[-1])

    return {
        "diagnosed_issue": result["diagnosed_issue"],
        "error_type": result["error_type"],
        "reasoning_trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: plan_recovery
# ---------------------------------------------------------------------------

def plan_recovery(
    state: HealingState,
    strategy_registry: dict[str, RecoveryStrategy],
    adaptive_healer: AdaptiveHealer,
) -> dict[str, Any]:
    """
    Score all registered strategies for the current error context, ask the LLM
    (or fallback) to pick the best one, and store the decision in state.
    """
    error_type = state.get("error_type", "unknown")
    error_context = state.get("error_signal", {})

    # Combine static strategy confidence with learned historical confidence
    combined_scores: dict[str, float] = {}
    for name, strategy in strategy_registry.items():
        static_score = strategy.can_handle(error_context)
        learned_score = adaptive_healer.score_action(error_type, name)
        # Weighted blend: 60% static suitability, 40% learned history
        combined_scores[name] = round(0.6 * static_score + 0.4 * learned_score, 4)

    # Sort descending and keep strategies above a noise floor
    available = [
        name for name, score in sorted(
            combined_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        if score > 0.1
    ]

    selection = select_strategy_with_llm(
        diagnosis=state.get("diagnosed_issue", ""),
        error_type=error_type,
        strategy_scores={k: combined_scores[k] for k in available},
    )

    selected = selection["selected_action"]
    confidence = combined_scores.get(selected, 0.0)

    # Fetch strategy parameters from the registry instance
    strategy_obj = strategy_registry.get(selected)
    action_params: dict[str, Any] = {}
    if strategy_obj:
        action_params = {
            k: v
            for k, v in vars(strategy_obj).items()
            if not k.startswith("_")
        }

    trace = list(state.get("reasoning_trace", []))
    trace.append(
        f"[plan_recovery] Selected '{selected}' (confidence={confidence:.2f}). "
        f"Reasoning: {selection['reasoning']}"
    )
    logger.info("plan_recovery: %s", trace[-1])

    return {
        "available_actions": available,
        "selected_action": selected,
        "action_parameters": action_params,
        "confidence_score": confidence,
        "reasoning_trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: execute
# ---------------------------------------------------------------------------

async def execute_recovery(
    state: HealingState,
    strategy_registry: dict[str, RecoveryStrategy],
) -> dict[str, Any]:
    """
    Invoke the selected strategy and store its result.
    """
    selected = state.get("selected_action", "")
    strategy = strategy_registry.get(selected)

    trace = list(state.get("reasoning_trace", []))

    if strategy is None:
        result = {
            "success": False,
            "message": f"Strategy '{selected}' not found in registry",
            "metadata": {},
        }
        trace.append(f"[execute] FAILED – strategy '{selected}' not found")
        logger.error("execute_recovery: %s", trace[-1])
        return {"execution_result": result, "reasoning_trace": trace}

    context = {
        **state.get("error_signal", {}),
        "error_signal": state.get("error_signal", {}),
        "attempt_count": state.get("attempt_count", 0),
        "action_parameters": state.get("action_parameters", {}),
    }

    try:
        result = await strategy.execute(context)
    except Exception as exc:  # pragma: no cover
        result = {
            "success": False,
            "message": f"Strategy '{selected}' raised an exception: {exc}",
            "metadata": {},
        }

    trace.append(
        f"[execute] Strategy '{selected}' → "
        f"success={result['success']}: {result['message']}"
    )
    logger.info("execute_recovery: %s", trace[-1])

    return {
        "execution_result": result,
        "attempt_count": state.get("attempt_count", 0) + 1,
        "reasoning_trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: validate
# ---------------------------------------------------------------------------

def validate_recovery(
    state: HealingState,
    adaptive_healer: AdaptiveHealer,
    confidence_threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Determine whether the pipeline has recovered.
    Records the outcome in the adaptive healer for future learning.
    """
    result = state.get("execution_result", {})
    success = result.get("success", False)
    selected = state.get("selected_action", "")
    error_type = state.get("error_type", "unknown")

    # Teach the adaptive healer
    adaptive_healer.record_outcome(error_type, selected, success)

    trace = list(state.get("reasoning_trace", []))
    if success:
        trace.append(
            f"[validate] Recovery SUCCESSFUL via '{selected}'. "
            f"Pipeline returning to healthy state."
        )
    else:
        trace.append(
            f"[validate] Recovery FAILED via '{selected}'. "
            f"Confidence={state.get('confidence_score', 0):.2f}, "
            f"threshold={confidence_threshold}"
        )
    logger.info("validate_recovery: %s", trace[-1])

    return {
        "is_recovered": success,
        "reasoning_trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: escalate
# ---------------------------------------------------------------------------

def escalate_to_human(state: HealingState) -> dict[str, Any]:
    """
    Signal that automated recovery has been exhausted and human intervention
    is required.  In production this would page the on-call team.
    """
    trace = list(state.get("reasoning_trace", []))
    trace.append(
        f"[escalate] Automated recovery exhausted for error type "
        f"'{state.get('error_type', 'unknown')}' "
        f"(last strategy: '{state.get('selected_action', 'none')}'). "
        "Escalating to human operators. "
        f"Attempts made: {state.get('attempt_count', 0)}"
    )
    logger.warning("escalate_to_human: %s", trace[-1])

    return {
        "is_recovered": False,
        "reasoning_trace": trace,
    }
