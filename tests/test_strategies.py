"""
Unit tests for RecoveryStrategy implementations.
"""

from __future__ import annotations

import asyncio

import pytest

from src.strategies import (
    CircuitBreakerStrategy,
    DataValidationStrategy,
    DeadLetterQueueStrategy,
    ResourceScalingStrategy,
    RetryStrategy,
    RollbackStrategy,
    build_strategy_registry,
)


# ---------------------------------------------------------------------------
# RetryStrategy
# ---------------------------------------------------------------------------

class TestRetryStrategy:
    def test_can_handle_timeout(self):
        s = RetryStrategy()
        assert s.can_handle({"type": "timeout"}) > 0.5

    def test_can_handle_low_for_unknown(self):
        s = RetryStrategy()
        assert s.can_handle({"type": "schema_mismatch"}) < 0.5

    def test_execute_first_attempt(self):
        s = RetryStrategy(initial_delay_ms=10)
        result = asyncio.run(s.execute({"attempt_count": 0}))
        assert result["success"] is True
        assert result["metadata"]["attempt"] == 1

    def test_execute_max_retries_exceeded(self):
        s = RetryStrategy(max_retries=2)
        result = asyncio.run(s.execute({"attempt_count": 2}))
        assert result["success"] is False

    def test_name(self):
        assert RetryStrategy.name == "retry"


# ---------------------------------------------------------------------------
# CircuitBreakerStrategy
# ---------------------------------------------------------------------------

class TestCircuitBreakerStrategy:
    def test_can_handle_service_unavailable(self):
        s = CircuitBreakerStrategy()
        assert s.can_handle({"type": "service_unavailable"}) > 0.5

    def test_can_handle_low_for_unrelated(self):
        s = CircuitBreakerStrategy()
        assert s.can_handle({"type": "timeout"}) < 0.5

    def test_execute_records_failures(self):
        # Reset class-level state to avoid cross-test contamination
        CircuitBreakerStrategy._failure_counts.clear()
        CircuitBreakerStrategy._open_since.clear()

        s = CircuitBreakerStrategy(failure_threshold=5)
        ctx = {"error_signal": {"source": "svc_cb_test"}}
        result = asyncio.run(s.execute(ctx))
        assert result["success"] is True
        assert result["metadata"]["failures"] == 1


# ---------------------------------------------------------------------------
# RollbackStrategy
# ---------------------------------------------------------------------------

class TestRollbackStrategy:
    def test_can_handle_data_corruption(self):
        s = RollbackStrategy()
        assert s.can_handle({"type": "data_corruption"}) > 0.5

    def test_execute_returns_success(self):
        s = RollbackStrategy()
        result = asyncio.run(s.execute({"error_signal": {"last_checkpoint": "cp_001"}}))
        assert result["success"] is True
        assert "cp_001" in result["message"]


# ---------------------------------------------------------------------------
# DataValidationStrategy
# ---------------------------------------------------------------------------

class TestDataValidationStrategy:
    def test_can_handle_dq_issue(self):
        s = DataValidationStrategy()
        assert s.can_handle({"type": "data_quality_issue"}) > 0.5

    def test_quarantines_records_above_threshold(self):
        s = DataValidationStrategy(null_threshold_pct=10.0)
        ctx = {
            "error_signal": {
                "null_pct": 20.0,
                "duplicate_pct": 0.0,
                "total_records": 1000,
            }
        }
        result = asyncio.run(s.execute(ctx))
        assert result["success"] is True
        assert result["metadata"]["quarantined_records"] > 0

    def test_no_quarantine_below_threshold(self):
        s = DataValidationStrategy(null_threshold_pct=10.0, duplicate_threshold_pct=5.0)
        ctx = {
            "error_signal": {
                "null_pct": 5.0,
                "duplicate_pct": 1.0,
                "total_records": 1000,
            }
        }
        result = asyncio.run(s.execute(ctx))
        assert result["metadata"]["quarantined_records"] == 0


# ---------------------------------------------------------------------------
# ResourceScalingStrategy
# ---------------------------------------------------------------------------

class TestResourceScalingStrategy:
    def test_can_handle_memory(self):
        s = ResourceScalingStrategy()
        assert s.can_handle({"type": "memory_exceeded"}) > 0.5

    def test_execute_scales(self):
        s = ResourceScalingStrategy(scale_factor=2)
        result = asyncio.run(s.execute({"attempt_count": 0}))
        assert result["success"] is True
        assert result["metadata"]["scale_factor"] == 2

    def test_execute_max_attempts_exceeded(self):
        s = ResourceScalingStrategy(max_scale_attempts=2)
        result = asyncio.run(s.execute({"attempt_count": 2}))
        assert result["success"] is False


# ---------------------------------------------------------------------------
# DeadLetterQueueStrategy
# ---------------------------------------------------------------------------

class TestDeadLetterQueueStrategy:
    def test_can_handle_unrecoverable(self):
        s = DeadLetterQueueStrategy()
        assert s.can_handle({"type": "unrecoverable_error"}) > 0.5

    def test_execute_enqueues(self):
        s = DeadLetterQueueStrategy(dlq_topic="test.dlq")
        result = asyncio.run(s.execute({"error_signal": {}}))
        assert result["success"] is True
        assert "test.dlq" in result["message"]


# ---------------------------------------------------------------------------
# build_strategy_registry
# ---------------------------------------------------------------------------

class TestBuildStrategyRegistry:
    def test_builds_known_strategies(self):
        config = [
            {"name": "retry", "parameters": {"max_retries": 5}},
            {"name": "rollback", "parameters": {}},
        ]
        registry = build_strategy_registry(config)
        assert "retry" in registry
        assert "rollback" in registry
        assert registry["retry"].max_retries == 5

    def test_skips_unknown_strategy(self):
        config = [{"name": "nonexistent_strategy", "parameters": {}}]
        registry = build_strategy_registry(config)
        assert "nonexistent_strategy" not in registry
