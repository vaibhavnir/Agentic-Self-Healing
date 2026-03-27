"""
FastAPI endpoint tests.
Uses FastAPI's synchronous TestClient; no external API keys required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Initialise app_state before importing app so the lifespan fires correctly
from api.dependencies import app_state
from api.app import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Return a TestClient that has run the lifespan (app is ready)."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["workflow_ready"] is True


# ---------------------------------------------------------------------------
# /heal (single)
# ---------------------------------------------------------------------------

class TestHealEndpoint:
    def test_heal_timeout(self, client):
        payload = {
            "error_signal": {
                "type": "timeout",
                "source": "postgres_loader",
                "message": "Query timed out",
                "last_checkpoint": "cp_api_001",
            }
        }
        resp = client.post("/heal", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "is_recovered" in data
        assert isinstance(data["is_recovered"], bool)
        assert data["error_type"] != ""
        assert len(data["reasoning_trace"]) >= 3

    def test_heal_data_quality_issue(self, client):
        payload = {
            "error_signal": {
                "type": "data_quality_issue",
                "source": "kafka_consumer",
                "message": "High null ratio",
                "null_pct": 20.0,
                "duplicate_pct": 1.0,
                "total_records": 5000,
                "last_checkpoint": "cp_api_002",
            }
        }
        resp = client.post("/heal", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_action"] != ""
        assert data["attempt_count"] >= 1

    def test_heal_missing_error_signal_returns_422(self, client):
        resp = client.post("/heal", json={})
        assert resp.status_code == 422

    def test_heal_unknown_error_handled_gracefully(self, client):
        payload = {"error_signal": {"type": "weird_unknown_error", "source": "x", "message": "?"}}
        resp = client.post("/heal", json=payload)
        assert resp.status_code == 200
        assert isinstance(resp.json()["is_recovered"], bool)

    def test_heal_response_schema(self, client):
        payload = {
            "error_signal": {
                "type": "schema_mismatch",
                "source": "s3_reader",
                "message": "Column type mismatch",
                "last_checkpoint": "cp_api_003",
            }
        }
        resp = client.post("/heal", json=payload)
        data = resp.json()
        expected_keys = {
            "is_recovered", "error_type", "diagnosed_issue", "selected_action",
            "confidence_score", "attempt_count", "execution_result",
            "reasoning_trace", "databricks_run_id",
        }
        assert expected_keys.issubset(data.keys())


# ---------------------------------------------------------------------------
# /heal/batch (parallel)
# ---------------------------------------------------------------------------

class TestHealBatchEndpoint:
    def test_batch_processes_multiple_failures(self, client):
        payload = {
            "failures": [
                {
                    "type": "timeout",
                    "source": "job_a",
                    "message": "Timed out",
                    "last_checkpoint": "cp_b_001",
                },
                {
                    "type": "data_quality_issue",
                    "source": "job_b",
                    "message": "Bad data",
                    "null_pct": 15.0,
                    "duplicate_pct": 0.0,
                    "total_records": 1000,
                    "last_checkpoint": "cp_b_002",
                },
                {
                    "type": "memory_exceeded",
                    "source": "job_c",
                    "message": "OOM",
                    "last_checkpoint": "cp_b_003",
                },
            ]
        }
        resp = client.post("/heal/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3
        assert data["recovered"] + data["escalated"] == 3

    def test_batch_empty_list_returns_422(self, client):
        resp = client.post("/heal/batch", json={"failures": []})
        assert resp.status_code == 422

    def test_batch_single_item(self, client):
        payload = {
            "failures": [
                {"type": "write_failure", "source": "s3_writer", "message": "S3 write failed", "last_checkpoint": "cp_b_004"}
            ]
        }
        resp = client.post("/heal/batch", json=payload)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

    def test_batch_result_contains_all_fields(self, client):
        payload = {
            "failures": [
                {"type": "timeout", "source": "test_src", "message": "timeout", "last_checkpoint": "cp_b_005"}
            ]
        }
        resp = client.post("/heal/batch", json=payload)
        result = resp.json()["results"][0]
        assert "is_recovered" in result
        assert "reasoning_trace" in result
        assert "selected_action" in result


# ---------------------------------------------------------------------------
# /outcomes
# ---------------------------------------------------------------------------

class TestOutcomesEndpoint:
    def test_outcomes_returns_list(self, client):
        resp = client.get("/outcomes")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # After the above tests have run there should be outcomes
        if data:
            row = data[0]
            assert "error_type" in row
            assert "action" in row
            assert "success_rate" in row
