"""
Unit tests for AdaptiveHealer.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.adaptive_healer import AdaptiveHealer


class TestAdaptiveHealer:
    def test_default_confidence_with_no_history(self):
        healer = AdaptiveHealer()
        assert healer.score_action("timeout", "retry") == 0.5

    def test_default_confidence_below_min_samples(self):
        healer = AdaptiveHealer(min_samples_for_confidence=5)
        for _ in range(4):
            healer.record_outcome("timeout", "retry", success=True)
        # Still below min_samples
        assert healer.score_action("timeout", "retry") == 0.5

    def test_confidence_calculated_after_min_samples(self):
        healer = AdaptiveHealer(min_samples_for_confidence=3)
        for _ in range(3):
            healer.record_outcome("timeout", "retry", success=True)
        assert healer.score_action("timeout", "retry") == pytest.approx(1.0)

    def test_mixed_outcomes_confidence(self):
        healer = AdaptiveHealer(min_samples_for_confidence=2)
        healer.record_outcome("timeout", "retry", success=True)
        healer.record_outcome("timeout", "retry", success=False)
        assert healer.score_action("timeout", "retry") == pytest.approx(0.5)

    def test_summary_returns_rows(self):
        healer = AdaptiveHealer(min_samples_for_confidence=1)
        healer.record_outcome("timeout", "retry", success=True)
        summary = healer.summary()
        assert len(summary) == 1
        assert summary[0]["error_type"] == "timeout"
        assert summary[0]["action"] == "retry"
        assert summary[0]["success_rate"] == pytest.approx(1.0)

    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "outcomes.json"
            healer = AdaptiveHealer(
                outcomes_file=str(path), min_samples_for_confidence=2
            )
            healer.record_outcome("timeout", "retry", success=True)
            healer.record_outcome("timeout", "retry", success=True)
            assert path.exists()

            # Reload
            healer2 = AdaptiveHealer(
                outcomes_file=str(path), min_samples_for_confidence=2
            )
            assert healer2.score_action("timeout", "retry") == pytest.approx(1.0)

    def test_load_missing_file_does_not_crash(self):
        healer = AdaptiveHealer(outcomes_file="/nonexistent/path/outcomes.json")
        assert healer.score_action("timeout", "retry") == 0.5
