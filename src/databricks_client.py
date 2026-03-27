"""
Databricks Jobs REST API client.

Wraps the v2.1 Jobs API to trigger job runs and query their status.

Environment variables
---------------------
DATABRICKS_HOST  – workspace URL, e.g. https://adb-1234567890.azuredatabricks.net
DATABRICKS_TOKEN – personal access token (PAT) or OAuth M2M token
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_JOBS_API = "/api/2.1/jobs"
_RUNS_API = "/api/2.1/jobs/runs"


class DatabricksJobsClient:
    """
    Async client for the Databricks Jobs API.

    Usage
    -----
    client = DatabricksJobsClient(host="https://...", token="dapi...")
    run_id = await client.trigger_run(job_id="123")
    status  = await client.get_run_status(run_id)
    await    client.cancel_run(run_id)
    """

    def __init__(self, host: str, token: str, timeout_s: float = 30.0) -> None:
        self._host = host.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._timeout = timeout_s

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "DatabricksJobsClient | None":
        """
        Return a client configured from ``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN``,
        or ``None`` when either variable is absent.
        """
        host = os.environ.get("DATABRICKS_HOST", "").strip()
        token = os.environ.get("DATABRICKS_TOKEN", "").strip()
        if not host or not token:
            return None
        return cls(host=host, token=token)

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    async def trigger_run(
        self,
        job_id: str | int,
        notebook_params: dict[str, str] | None = None,
        python_params: list[str] | None = None,
    ) -> str:
        """
        Trigger a job run and return the ``run_id`` as a string.

        Parameters
        ----------
        job_id:
            Databricks job ID.
        notebook_params:
            Key/value pairs passed to notebook tasks as widget values.
        python_params:
            Command-line parameters passed to Python script tasks.
        """
        payload: dict[str, Any] = {"job_id": int(job_id)}
        if notebook_params:
            payload["notebook_params"] = notebook_params
        if python_params:
            payload["python_params"] = python_params

        async with httpx.AsyncClient(
            headers=self._headers, timeout=self._timeout
        ) as client:
            resp = await client.post(
                f"{self._host}{_RUNS_API}/now",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        run_id = str(data["run_id"])
        logger.info("DatabricksJobsClient: triggered job_id=%s → run_id=%s", job_id, run_id)
        return run_id

    async def get_run_status(self, run_id: str | int) -> dict[str, Any]:
        """
        Return the run state dict for the given ``run_id``.

        The dict contains at minimum:
          - ``life_cycle_state``: PENDING / RUNNING / TERMINATING / TERMINATED / SKIPPED / INTERNAL_ERROR
          - ``result_state`` (when terminated): SUCCESS / FAILED / TIMEDOUT / CANCELED / MAXIMUM_CONCURRENT_RUNS_REACHED
          - ``state_message``: human-readable description
        """
        async with httpx.AsyncClient(
            headers=self._headers, timeout=self._timeout
        ) as client:
            resp = await client.get(
                f"{self._host}{_RUNS_API}/get",
                params={"run_id": int(run_id)},
            )
            resp.raise_for_status()
            data = resp.json()

        state = data.get("state", {})
        logger.debug(
            "DatabricksJobsClient: run_id=%s state=%s", run_id, state
        )
        return state

    async def cancel_run(self, run_id: str | int) -> None:
        """Cancel an active run."""
        async with httpx.AsyncClient(
            headers=self._headers, timeout=self._timeout
        ) as client:
            resp = await client.post(
                f"{self._host}{_RUNS_API}/cancel",
                json={"run_id": int(run_id)},
            )
            resp.raise_for_status()

        logger.info("DatabricksJobsClient: cancelled run_id=%s", run_id)

    async def list_jobs(self, name_filter: str | None = None) -> list[dict[str, Any]]:
        """
        Return a list of jobs in the workspace.
        Optionally filter by job name substring.
        """
        params: dict[str, Any] = {"limit": 25}
        if name_filter:
            params["name"] = name_filter

        jobs: list[dict[str, Any]] = []
        async with httpx.AsyncClient(
            headers=self._headers, timeout=self._timeout
        ) as client:
            while True:
                resp = await client.get(
                    f"{self._host}{_JOBS_API}/list",
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
                jobs.extend(data.get("jobs", []))
                if not data.get("has_more", False):
                    break
                params["page_token"] = data["next_page_token"]

        return jobs
