"""
App-level singleton state: compiled workflow + AdaptiveHealer.

Initialized once during FastAPI lifespan and shared across all requests.
Thread safety is ensured by the AdaptiveHealer's internal lock; the compiled
LangGraph workflow is stateless (each invocation creates its own HealingState)
so it is safe to share without additional locking.
"""

from __future__ import annotations

import os
from typing import Any

from src.adaptive_healer import AdaptiveHealer
from src.config_loader import load_config
from src.workflow import create_self_healing_workflow


class AppState:
    """Holds the single shared workflow and healer instances."""

    def __init__(self) -> None:
        self.workflow: Any = None
        self.healer: AdaptiveHealer | None = None
        self._ready: bool = False

    @property
    def ready(self) -> bool:
        return self._ready

    def initialise(self, config_path: str | None = None) -> None:
        """
        Load configuration and build the workflow.
        Call once at application startup.
        """
        config_path = config_path or os.environ.get("HEALING_CONFIG")
        config = load_config(config_path)

        strategy_config = config.get("strategies", [])
        feedback_cfg = config.get("feedback", {})
        escalation_cfg = config.get("escalation", {})

        outcomes_file = feedback_cfg.get("outcomes_file", "data/healing_outcomes.json")
        min_samples = feedback_cfg.get("min_samples_for_confidence", 5)
        confidence_threshold = escalation_cfg.get("confidence_threshold", 0.3)

        self.healer = AdaptiveHealer(
            outcomes_file=outcomes_file,
            min_samples_for_confidence=min_samples,
        )
        self.workflow = create_self_healing_workflow(
            strategy_config=strategy_config,
            adaptive_healer=self.healer,
            confidence_threshold=confidence_threshold,
        )
        self._ready = True


# Module-level singleton – shared by all FastAPI route handlers
app_state = AppState()
