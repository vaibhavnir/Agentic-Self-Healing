"""
Adaptive healer: tracks historical success/failure rates per (error_type, action)
pair and uses them to adjust confidence scores dynamically.

Outcomes are optionally persisted to disk so the system learns across restarts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIDENCE = 0.5


class AdaptiveHealer:
    """
    Learns from past healing attempts and adjusts per-action confidence scores.

    Usage
    -----
    healer = AdaptiveHealer(outcomes_file="data/healing_outcomes.json")
    score = healer.score_action("timeout", "retry")
    healer.record_outcome("timeout", "retry", success=True)
    """

    def __init__(
        self,
        outcomes_file: str | None = None,
        min_samples_for_confidence: int = 5,
    ) -> None:
        self.min_samples = min_samples_for_confidence
        self.outcomes_file = Path(outcomes_file) if outcomes_file else None

        # { (error_type, action): {"successes": int, "total": int} }
        self._history: dict[tuple[str, str], dict[str, int]] = {}
        # Protects _history and file I/O from concurrent access
        self._lock = threading.Lock()

        if self.outcomes_file and self.outcomes_file.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_action(self, error_type: str, action: str) -> float:
        """
        Return a confidence score in [0, 1].

        If fewer than `min_samples` outcomes are recorded the default (0.5)
        is returned so the system doesn't over-fit on sparse data.
        """
        with self._lock:
            key = (error_type, action)
            record = self._history.get(key)
            if record is None or record["total"] < self.min_samples:
                return _DEFAULT_CONFIDENCE

            return record["successes"] / record["total"]

    def record_outcome(self, error_type: str, action: str, success: bool) -> None:
        """Store the outcome of a healing attempt and (optionally) persist it."""
        with self._lock:
            key = (error_type, action)
            if key not in self._history:
                self._history[key] = {"successes": 0, "total": 0}

            self._history[key]["total"] += 1
            if success:
                self._history[key]["successes"] += 1

            logger.debug(
                "AdaptiveHealer: recorded %s for (%s, %s) → %s",
                "SUCCESS" if success else "FAILURE",
                error_type,
                action,
                self._history[key],
            )

            if self.outcomes_file:
                self._persist()

    def summary(self) -> list[dict[str, Any]]:
        """Return a human-readable summary of all tracked outcomes."""
        with self._lock:
            rows = []
            for (error_type, action), record in self._history.items():
                total = record["total"]
                successes = record["successes"]
                rows.append(
                    {
                        "error_type": error_type,
                        "action": action,
                        "successes": successes,
                        "failures": total - successes,
                        "total": total,
                        "success_rate": round(successes / total, 3) if total else 0,
                    }
                )
        return sorted(rows, key=lambda r: r["success_rate"], reverse=True)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    # _persist and _load are always called while self._lock is already held.

    def _persist(self) -> None:
        assert self.outcomes_file is not None
        self.outcomes_file.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            f"{et}::{act}": rec for (et, act), rec in self._history.items()
        }
        try:
            self.outcomes_file.write_text(
                json.dumps(serialisable, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("AdaptiveHealer: could not persist outcomes: %s", exc)

    def _load(self) -> None:
        assert self.outcomes_file is not None
        try:
            data: dict[str, dict[str, int]] = json.loads(
                self.outcomes_file.read_text(encoding="utf-8")
            )
            for compound_key, record in data.items():
                if "::" not in compound_key:
                    logger.warning(
                        "AdaptiveHealer: skipping malformed key '%s' "
                        "(expected format 'error_type::action')",
                        compound_key,
                    )
                    continue
                et, act = compound_key.split("::", 1)
                self._history[(et, act)] = record
            logger.info(
                "AdaptiveHealer: loaded %d outcome records from '%s'",
                len(self._history),
                self.outcomes_file,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("AdaptiveHealer: could not load outcomes: %s", exc)
