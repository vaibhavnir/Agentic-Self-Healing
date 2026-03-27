"""
Recovery strategy library.

Each strategy implements the RecoveryStrategy ABC so the engine can:
  1. Ask whether the strategy is applicable (can_handle → confidence float)
  2. Execute the strategy and return a result dict
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class RecoveryStrategy(ABC):
    """Base class for all recovery strategies."""

    name: str = "base"

    @abstractmethod
    def can_handle(self, error_context: dict[str, Any]) -> float:
        """
        Return a confidence score in [0, 1] indicating how well this strategy
        applies to the given error context.  0 means 'not applicable'.
        """

    @abstractmethod
    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the recovery action.

        Returns a result dict with at minimum:
          - success (bool)
          - message (str)
          - metadata (dict) – any strategy-specific details
        """


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class RetryStrategy(RecoveryStrategy):
    """Retry the failed operation with exponential back-off."""

    name = "retry"

    def __init__(
        self,
        max_retries: int = 3,
        backoff_multiplier: float = 2.0,
        initial_delay_ms: int = 1000,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier
        self.initial_delay_s = initial_delay_ms / 1000.0

    def can_handle(self, error_context: dict[str, Any]) -> float:
        retriable = {"timeout", "connection_refused", "transient_error"}
        error_type = error_context.get("type", "")
        return 0.85 if error_type in retriable else 0.15

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        attempt = context.get("attempt_count", 0)
        if attempt >= self.max_retries:
            return {
                "success": False,
                "message": f"Max retries ({self.max_retries}) exceeded",
                "metadata": {"attempt": attempt},
            }

        delay = self.initial_delay_s * (self.backoff_multiplier ** attempt)
        logger.info("RetryStrategy: waiting %.2fs before retry #%d", delay, attempt + 1)
        await asyncio.sleep(delay)

        # Simulate retry – in production this would re-invoke the ETL step
        return {
            "success": True,
            "message": f"Retry #{attempt + 1} scheduled after {delay:.2f}s back-off",
            "metadata": {"delay_s": delay, "attempt": attempt + 1},
        }


class CircuitBreakerStrategy(RecoveryStrategy):
    """
    Open the circuit to stop cascading failures, then allow a probe after
    the recovery timeout.
    """

    name = "circuit_breaker"

    # Shared state across instances (class-level)
    _failure_counts: dict[str, int] = {}
    _open_since: dict[str, float] = {}

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: int = 300,
        half_open_max_calls: int = 3,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_max_calls = half_open_max_calls

    def can_handle(self, error_context: dict[str, Any]) -> float:
        applicable = {"service_unavailable", "rate_limit_exceeded"}
        return 0.80 if error_context.get("type", "") in applicable else 0.10

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        service = context.get("error_signal", {}).get("source", "unknown_service")
        now = time.time()

        failures = self._failure_counts.get(service, 0) + 1
        self._failure_counts[service] = failures

        if failures >= self.failure_threshold:
            open_since = self._open_since.get(service, now)
            self._open_since[service] = open_since
            elapsed = now - open_since

            if elapsed >= self.recovery_timeout_s:
                # Half-open: allow probe
                self._failure_counts[service] = 0
                del self._open_since[service]
                return {
                    "success": True,
                    "message": "Circuit half-opened – probe request allowed",
                    "metadata": {"service": service, "elapsed_s": elapsed},
                }

            return {
                "success": False,
                "message": (
                    f"Circuit OPEN for '{service}'. "
                    f"Will attempt recovery in {self.recovery_timeout_s - elapsed:.0f}s"
                ),
                "metadata": {
                    "service": service,
                    "failures": failures,
                    "open_since": open_since,
                },
            }

        return {
            "success": True,
            "message": f"Failure #{failures} recorded for '{service}' (threshold={self.failure_threshold})",
            "metadata": {"service": service, "failures": failures},
        }


class RollbackStrategy(RecoveryStrategy):
    """Roll back to the last known-good checkpoint."""

    name = "rollback"

    def __init__(self, keep_snapshots: int = 5, checkpoint_interval_s: int = 60) -> None:
        self.keep_snapshots = keep_snapshots
        self.checkpoint_interval_s = checkpoint_interval_s

    def can_handle(self, error_context: dict[str, Any]) -> float:
        rollback_errors = {"data_corruption", "schema_mismatch", "write_failure"}
        return 0.90 if error_context.get("type", "") in rollback_errors else 0.05

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        checkpoint = context.get("error_signal", {}).get("last_checkpoint", "unknown")
        logger.info("RollbackStrategy: rolling back to checkpoint '%s'", checkpoint)
        # In production: restore state from checkpoint store
        return {
            "success": True,
            "message": f"Pipeline rolled back to checkpoint '{checkpoint}'",
            "metadata": {"checkpoint": checkpoint},
        }


class DataValidationStrategy(RecoveryStrategy):
    """Quarantine bad records and continue with clean data."""

    name = "data_validation"

    def __init__(
        self,
        null_threshold_pct: float = 10.0,
        duplicate_threshold_pct: float = 5.0,
        schema_strict: bool = True,
    ) -> None:
        self.null_threshold_pct = null_threshold_pct
        self.duplicate_threshold_pct = duplicate_threshold_pct
        self.schema_strict = schema_strict

    def can_handle(self, error_context: dict[str, Any]) -> float:
        dq_errors = {"data_quality_issue", "null_values_exceeded", "type_mismatch"}
        return 0.88 if error_context.get("type", "") in dq_errors else 0.10

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        signal = context.get("error_signal", {})
        null_pct = signal.get("null_pct", 0.0)
        dup_pct = signal.get("duplicate_pct", 0.0)

        quarantined = 0
        issues = []

        if null_pct > self.null_threshold_pct:
            quarantined += int(signal.get("total_records", 0) * null_pct / 100)
            issues.append(f"null_pct={null_pct:.1f}% exceeds threshold {self.null_threshold_pct}%")

        if dup_pct > self.duplicate_threshold_pct:
            quarantined += int(signal.get("total_records", 0) * dup_pct / 100)
            issues.append(f"dup_pct={dup_pct:.1f}% exceeds threshold {self.duplicate_threshold_pct}%")

        return {
            "success": True,
            "message": "Data validation completed – bad records quarantined",
            "metadata": {
                "quarantined_records": quarantined,
                "issues": issues,
                "schema_strict": self.schema_strict,
            },
        }


class ResourceScalingStrategy(RecoveryStrategy):
    """Request additional compute resources when the pipeline is resource-starved."""

    name = "resource_scaling"

    def __init__(self, scale_factor: int = 2, max_scale_attempts: int = 3) -> None:
        self.scale_factor = scale_factor
        self.max_scale_attempts = max_scale_attempts

    def can_handle(self, error_context: dict[str, Any]) -> float:
        resource_errors = {"memory_exceeded", "cpu_throttled", "disk_full"}
        return 0.82 if error_context.get("type", "") in resource_errors else 0.05

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        attempt = context.get("attempt_count", 0)
        if attempt >= self.max_scale_attempts:
            return {
                "success": False,
                "message": "Max scaling attempts reached",
                "metadata": {"attempt": attempt},
            }
        scale = self.scale_factor * (attempt + 1)
        logger.info("ResourceScalingStrategy: requesting %dx resources", scale)
        return {
            "success": True,
            "message": f"Resource scaling request submitted (×{scale})",
            "metadata": {"scale_factor": scale, "attempt": attempt + 1},
        }


class DeadLetterQueueStrategy(RecoveryStrategy):
    """Send unrecoverable messages to a dead-letter queue and alert the team."""

    name = "dead_letter_queue"

    def __init__(self, dlq_topic: str = "etl.dead_letter", alert_on_enqueue: bool = True) -> None:
        self.dlq_topic = dlq_topic
        self.alert_on_enqueue = alert_on_enqueue

    def can_handle(self, error_context: dict[str, Any]) -> float:
        dlq_errors = {"unrecoverable_error", "max_retries_exceeded"}
        return 0.95 if error_context.get("type", "") in dlq_errors else 0.02

    async def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        signal = context.get("error_signal", {})
        logger.warning(
            "DeadLetterQueueStrategy: enqueuing to '%s'. Alert=%s",
            self.dlq_topic,
            self.alert_on_enqueue,
        )
        return {
            "success": True,
            "message": f"Event enqueued to dead-letter topic '{self.dlq_topic}'",
            "metadata": {
                "topic": self.dlq_topic,
                "alert_sent": self.alert_on_enqueue,
                "original_error": signal,
            },
        }


# ---------------------------------------------------------------------------
# Strategy registry helper
# ---------------------------------------------------------------------------

def build_strategy_registry(config: list[dict[str, Any]]) -> dict[str, RecoveryStrategy]:
    """
    Build a mapping of strategy name → instance from the YAML config list.
    Only strategies explicitly listed in the config are registered.
    """
    _class_map: dict[str, type[RecoveryStrategy]] = {
        "retry": RetryStrategy,
        "circuit_breaker": CircuitBreakerStrategy,
        "rollback": RollbackStrategy,
        "data_validation": DataValidationStrategy,
        "resource_scaling": ResourceScalingStrategy,
        "dead_letter_queue": DeadLetterQueueStrategy,
    }

    registry: dict[str, RecoveryStrategy] = {}
    for entry in config:
        name = entry["name"]
        cls = _class_map.get(name)
        if cls is None:
            logger.warning("Unknown strategy '%s' in config – skipped", name)
            continue
        params = entry.get("parameters", {})
        registry[name] = cls(**params)

    return registry
