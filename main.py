"""
Entry point for the Agentic Self-Healing ETL pipeline.

Usage
-----
    python main.py                          # run built-in demo scenarios
    python main.py --config path/to/cfg.yaml

Environment variables
---------------------
    OPENAI_API_KEY   – enables LLM-powered diagnosis and strategy selection.
                       When absent, the system falls back to config-driven rules.
    HEALING_CONFIG   – alternative path to healing_strategies.yaml
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from src.adaptive_healer import AdaptiveHealer
from src.config_loader import load_config
from src.state import HealingState
from src.workflow import create_self_healing_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demo error scenarios
# ---------------------------------------------------------------------------

DEMO_SCENARIOS: list[dict[str, Any]] = [
    {
        "type": "timeout",
        "source": "postgres_loader",
        "message": "Query timed out after 30s",
        "last_checkpoint": "checkpoint_2024_01_15_09_00",
    },
    {
        "type": "data_quality_issue",
        "source": "kafka_consumer",
        "message": "15% null values in column 'user_id'",
        "null_pct": 15.0,
        "duplicate_pct": 2.0,
        "total_records": 50_000,
        "last_checkpoint": "checkpoint_2024_01_15_09_30",
    },
    {
        "type": "service_unavailable",
        "source": "downstream_api",
        "message": "HTTP 503 from reporting service",
        "last_checkpoint": "checkpoint_2024_01_15_10_00",
    },
    {
        "type": "schema_mismatch",
        "source": "s3_parquet_reader",
        "message": "Column 'amount' expected DECIMAL got STRING",
        "last_checkpoint": "checkpoint_2024_01_15_10_15",
    },
    {
        "type": "memory_exceeded",
        "source": "spark_aggregation",
        "message": "OOM error in shuffle partition 42",
        "last_checkpoint": "checkpoint_2024_01_15_11_00",
    },
]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

async def heal(
    error_signal: dict[str, Any],
    workflow: Any,
) -> HealingState:
    """
    Invoke the self-healing workflow for a single error signal and return
    the final state.
    """
    initial_state: HealingState = {
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

    final_state: HealingState = await workflow.ainvoke(initial_state)
    return final_state


def _print_result(scenario_idx: int, state: HealingState) -> None:
    """Pretty-print the healing outcome for a single scenario."""
    status = "✅ RECOVERED" if state.get("is_recovered") else "❌ ESCALATED"
    print(f"\n{'=' * 65}")
    print(f"  Scenario {scenario_idx + 1}: {status}")
    print(f"{'=' * 65}")
    print(f"  Error type   : {state.get('error_type')}")
    print(f"  Diagnosis    : {state.get('diagnosed_issue')}")
    print(f"  Strategy     : {state.get('selected_action')}")
    print(f"  Confidence   : {state.get('confidence_score', 0):.2f}")
    print(f"  Attempts     : {state.get('attempt_count', 0)}")
    result = state.get("execution_result", {})
    print(f"  Result msg   : {result.get('message', '-')}")
    print("\n  Reasoning trace:")
    for step in state.get("reasoning_trace", []):
        print(f"    {step}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    config_path = os.environ.get("HEALING_CONFIG") or (
        sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--config" else None
    )
    config = load_config(config_path)

    strategy_config = config.get("strategies", [])
    feedback_cfg = config.get("feedback", {})
    escalation_cfg = config.get("escalation", {})

    outcomes_file = feedback_cfg.get("outcomes_file", "data/healing_outcomes.json")
    min_samples = feedback_cfg.get("min_samples_for_confidence", 5)
    confidence_threshold = escalation_cfg.get("confidence_threshold", 0.3)

    adaptive_healer = AdaptiveHealer(
        outcomes_file=outcomes_file,
        min_samples_for_confidence=min_samples,
    )

    workflow = create_self_healing_workflow(
        strategy_config=strategy_config,
        adaptive_healer=adaptive_healer,
        confidence_threshold=confidence_threshold,
    )

    print("\nAgentic Self-Healing ETL Pipeline – Demo Run")
    print("=" * 65)

    for idx, scenario in enumerate(DEMO_SCENARIOS):
        logger.info("--- Running scenario %d/%d ---", idx + 1, len(DEMO_SCENARIOS))
        final_state = await heal(scenario, workflow)
        _print_result(idx, final_state)

    print("\n\nAdaptive Healer – Outcome Summary")
    print("-" * 65)
    summary = adaptive_healer.summary()
    if summary:
        for row in summary:
            print(
                f"  {row['error_type']:30s} | {row['action']:25s} | "
                f"success_rate={row['success_rate']:.2f} "
                f"({row['successes']}/{row['total']})"
            )
    else:
        print("  (no outcomes recorded yet)")


if __name__ == "__main__":
    asyncio.run(main())
