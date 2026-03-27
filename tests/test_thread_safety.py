"""
Thread-safety tests for AdaptiveHealer and CircuitBreakerStrategy.

Runs many concurrent operations and checks that no data is lost or corrupted.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from src.adaptive_healer import AdaptiveHealer
from src.strategies import CircuitBreakerStrategy


# ---------------------------------------------------------------------------
# AdaptiveHealer thread safety
# ---------------------------------------------------------------------------

class TestAdaptiveHealerThreadSafety:
    def test_concurrent_record_outcome_no_lost_writes(self):
        """
        100 threads each record 10 outcomes.
        Total must equal 1000 with no lost writes.
        """
        healer = AdaptiveHealer(min_samples_for_confidence=1)
        n_threads = 100
        records_each = 10

        def _record():
            for _ in range(records_each):
                healer.record_outcome("timeout", "retry", success=True)

        threads = [threading.Thread(target=_record) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        score = healer.score_action("timeout", "retry")
        # All successes → rate must be 1.0
        assert score == pytest.approx(1.0)
        # Total count must be exactly n_threads * records_each
        assert healer._history[("timeout", "retry")]["total"] == n_threads * records_each

    def test_concurrent_mixed_outcomes(self):
        """
        50 threads record success, 50 record failure.
        Success rate must be 0.5 with no lost writes.
        """
        healer = AdaptiveHealer(min_samples_for_confidence=1)
        n_each = 50

        def _success():
            for _ in range(10):
                healer.record_outcome("service_unavailable", "circuit_breaker", success=True)

        def _failure():
            for _ in range(10):
                healer.record_outcome("service_unavailable", "circuit_breaker", success=False)

        threads = (
            [threading.Thread(target=_success) for _ in range(n_each)]
            + [threading.Thread(target=_failure) for _ in range(n_each)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        record = healer._history[("service_unavailable", "circuit_breaker")]
        assert record["total"] == n_each * 10 * 2
        assert record["successes"] == n_each * 10

    def test_concurrent_score_and_record(self):
        """
        score_action and record_outcome called simultaneously must not deadlock.
        """
        healer = AdaptiveHealer(min_samples_for_confidence=1)
        results: list[float] = []
        lock = threading.Lock()

        def _reader():
            for _ in range(50):
                score = healer.score_action("timeout", "retry")
                with lock:
                    results.append(score)

        def _writer():
            for i in range(50):
                healer.record_outcome("timeout", "retry", success=(i % 2 == 0))

        threads = (
            [threading.Thread(target=_reader) for _ in range(5)]
            + [threading.Thread(target=_writer) for _ in range(5)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Simply asserting no deadlock/exception occurred and results are valid floats
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_concurrent_persist_and_load(self, tmp_path):
        """
        Multiple threads persisting outcomes to the same file must not corrupt it.
        """
        outcomes_file = str(tmp_path / "outcomes.json")
        healer = AdaptiveHealer(outcomes_file=outcomes_file, min_samples_for_confidence=1)

        def _write():
            for i in range(20):
                healer.record_outcome(f"error_{i % 5}", "retry", success=bool(i % 2))

        threads = [threading.Thread(target=_write) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Reload and verify the file is valid JSON
        healer2 = AdaptiveHealer(outcomes_file=outcomes_file, min_samples_for_confidence=1)
        assert len(healer2._history) > 0


# ---------------------------------------------------------------------------
# CircuitBreakerStrategy thread safety
# ---------------------------------------------------------------------------

class TestCircuitBreakerThreadSafety:
    def setup_method(self):
        # Reset class-level state before each test
        CircuitBreakerStrategy._failure_counts.clear()
        CircuitBreakerStrategy._open_since.clear()

    def test_concurrent_execute_no_negative_counts(self):
        """
        100 concurrent coroutines recording failures must never produce
        negative failure counts or corrupt state.
        """
        strategy = CircuitBreakerStrategy(failure_threshold=200)

        async def _run():
            await strategy.execute({"error_signal": {"source": "svc_concurrent"}})

        async def _main():
            await asyncio.gather(*[_run() for _ in range(100)])

        asyncio.run(_main())

        count = CircuitBreakerStrategy._failure_counts.get("svc_concurrent", 0)
        assert count == 100

    def test_concurrent_different_services_isolated(self):
        """
        Concurrent failures for different services must not affect each other's counts.
        """
        strategy = CircuitBreakerStrategy(failure_threshold=200)
        services = [f"svc_{i}" for i in range(10)]

        async def _run(svc):
            await strategy.execute({"error_signal": {"source": svc}})

        async def _main():
            tasks = [_run(svc) for svc in services for _ in range(5)]
            await asyncio.gather(*tasks)

        asyncio.run(_main())

        for svc in services:
            assert CircuitBreakerStrategy._failure_counts.get(svc, 0) == 5

    def test_concurrent_threads_and_coroutines(self):
        """
        Mix threading and asyncio to ensure the threading.Lock protects
        against cross-thread mutations.
        """
        strategy = CircuitBreakerStrategy(failure_threshold=1000)
        errors: list[Exception] = []

        def _thread_run():
            try:
                asyncio.run(strategy.execute({"error_signal": {"source": "mixed_svc"}}))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_thread_run) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Exceptions in threads: {errors}"
        assert CircuitBreakerStrategy._failure_counts.get("mixed_svc", 0) == 20
