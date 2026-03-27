"""
LLM interface for the self-healing workflow.

Provides a thin wrapper that can use either:
  - A real LangChain LLM (when OPENAI_API_KEY / ANTHROPIC_API_KEY is set), or
  - A rule-based fallback that works without any API key.

The fallback is intentionally transparent – it maps error types to diagnoses
using the config rather than hard-coded if/else chains.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DIAGNOSIS_PROMPT = """\
You are an expert ETL pipeline reliability engineer.

Analyse the following error signal and provide:
1. A concise diagnosis (one sentence).
2. The canonical error_type key (choose one of the known types listed below).
3. A brief explanation of the root cause.

Known error types: {known_types}

Error signal:
{error_signal}

Respond in JSON with keys: diagnosed_issue, error_type, root_cause_explanation
"""

STRATEGY_SELECTION_PROMPT = """\
You are choosing a recovery strategy for an ETL pipeline failure.

Diagnosis: {diagnosis}
Error type: {error_type}
Available strategies (with confidence scores): {strategy_scores}

Select the single best strategy name and explain why in one sentence.

Respond in JSON with keys: selected_action, reasoning
"""


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _try_real_llm(prompt: str) -> str | None:
    """
    Attempt to call a real LLM.  Returns the raw text response or None if
    no LLM is configured.
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
        from langchain_core.messages import HumanMessage  # type: ignore

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as exc:  # pragma: no cover
        logger.debug("Real LLM call failed: %s", exc)
        return None


def _parse_json_response(text: str) -> dict[str, Any]:
    """
    Extract JSON from an LLM response that may include surrounding markdown
    code fences or plain text.
    """
    import json
    import re

    # Strip markdown fences
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    candidate = match.group(1) if match else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM JSON response: %s", text[:200])
        return {}


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def diagnose_with_llm(
    error_signal: dict[str, Any],
    known_types: list[str],
) -> dict[str, str]:
    """
    Return {"diagnosed_issue": ..., "error_type": ..., "root_cause_explanation": ...}.

    Falls back to a config-driven heuristic when no LLM is available.
    """
    prompt = DIAGNOSIS_PROMPT.format(
        known_types=", ".join(known_types),
        error_signal=error_signal,
    )

    raw = _try_real_llm(prompt)
    if raw:
        result = _parse_json_response(raw)
        if result.get("error_type"):
            return result

    # --- Rule-based fallback (config-driven, not hard-coded) ---
    error_type = error_signal.get("type", "unknown")
    if error_type not in known_types:
        # Fuzzy match: pick the closest known type by shared substring
        for known in known_types:
            if known in error_type or error_type in known:
                error_type = known
                break

    return {
        "diagnosed_issue": (
            f"ETL pipeline failure: {error_signal.get('message', 'no details')}"
        ),
        "error_type": error_type,
        "root_cause_explanation": (
            f"Error type '{error_type}' detected in component "
            f"'{error_signal.get('source', 'unknown')}'"
        ),
    }


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def select_strategy_with_llm(
    diagnosis: str,
    error_type: str,
    strategy_scores: dict[str, float],
) -> dict[str, str]:
    """
    Return {"selected_action": ..., "reasoning": ...}.

    Falls back to the highest-scoring strategy when no LLM is available.
    """
    if not strategy_scores:
        return {"selected_action": "dead_letter_queue", "reasoning": "No strategies available"}

    prompt = STRATEGY_SELECTION_PROMPT.format(
        diagnosis=diagnosis,
        error_type=error_type,
        strategy_scores=strategy_scores,
    )

    raw = _try_real_llm(prompt)
    if raw:
        result = _parse_json_response(raw)
        if result.get("selected_action") in strategy_scores:
            return result

    # Fallback: pick highest combined score
    best = max(strategy_scores, key=lambda k: strategy_scores[k])
    return {
        "selected_action": best,
        "reasoning": (
            f"Strategy '{best}' selected with highest combined confidence "
            f"({strategy_scores[best]:.2f}) for error type '{error_type}'"
        ),
    }
