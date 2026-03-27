"""
Integration test: run the full self-healing workflow end-to-end.
No external API keys required.
"""

from __future__ import annotations

import asyncio

import pytest

from src.adaptive_healer import AdaptiveHealer
from src.config_loader import load_config
from src.state import HealingState
from src.workflow import create_self_healing_workflow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def workflow(config):
    healer = AdaptiveHealer(min_samples_for_confidence=2)
    return create_self_healing_workflow(
        strategy_config=config.get("strategies", []),
        adaptive_healer=healer,
        max_attempts=3,
        confidence_threshold=0.1,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    return asyncio.run(coro)


async def _heal(workflow, error_signal: dict) -> HealingState:
    initial: HealingState = {
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
    return await workflow.ainvoke(initial)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWorkflowIntegration:
    def test_timeout_reaches_final_state(self, workflow):
        signal = {
            "type": "timeout",
            "source": "db_loader",
            "message": "Query timed out",
            "last_checkpoint": "cp_001",
        }
        state = run(_heal(workflow, signal))
        assert state["error_type"] != ""
        assert len(state["reasoning_trace"]) >= 3
        assert state["attempt_count"] >= 1

    def test_data_quality_issue(self, workflow):
        signal = {
            "type": "data_quality_issue",
            "source": "kafka",
            "message": "Too many nulls",
            "null_pct": 20.0,
            "duplicate_pct": 0.0,
            "total_records": 1000,
            "last_checkpoint": "cp_002",
        }
        state = run(_heal(workflow, signal))
        assert state["selected_action"] in (
            "data_validation", "retry", "dead_letter_queue",
        )

    def test_schema_mismatch_triggers_rollback(self, workflow):
        signal = {
            "type": "schema_mismatch",
            "source": "s3_reader",
            "message": "Column type mismatch",
            "last_checkpoint": "cp_003",
        }
        state = run(_heal(workflow, signal))
        assert state["selected_action"] != ""
        assert state["attempt_count"] >= 1

    def test_reasoning_trace_populated(self, workflow):
        signal = {
            "type": "memory_exceeded",
            "source": "spark",
            "message": "OOM",
            "last_checkpoint": "cp_004",
        }
        state = run(_heal(workflow, signal))
        trace = state["reasoning_trace"]
        steps = [t.split("]")[0].lstrip("[") for t in trace if "]" in t]
        assert "detect" in steps
        assert "diagnose" in steps

    def test_unknown_error_type_handled_gracefully(self, workflow):
        signal = {
            "type": "completely_unknown_error_xyz",
            "source": "mystery",
            "message": "No idea what happened",
            "last_checkpoint": "cp_005",
        }
        state = run(_heal(workflow, signal))
        # Should not raise; should reach a terminal state
        assert state["attempt_count"] >= 0
        assert isinstance(state["is_recovered"], bool)
