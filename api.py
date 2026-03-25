"""
api.py
------
FastAPI webhook server that receives Databricks job failure notifications and
triggers the LangGraph self-healing workflow.

Endpoints
---------
POST /webhook/job-failure
    Accepts a JSON payload describing a failed Databricks job and immediately
    launches the ``run_healing_workflow`` coroutine in the background.

GET /health
    Simple liveness probe used by load-balancers and Kubernetes health checks.

Running locally
---------------
::

    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Databricks Notification Integration
------------------------------------
Configure a Databricks job notification to POST to this webhook on failure:

1. In the Databricks UI, open your job → *Edit* → *Notifications*.
2. Add a *Webhook* destination pointing to
   ``https://<your-host>/webhook/job-failure``.
3. Databricks will POST a JSON body similar to::

       {
         "job_id": "98765",
         "run_id": "112233",
         "event_type": "job_failure",
         "error": "AnalysisException: A schema mismatch detected..."
       }
"""

import asyncio
import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_workflow import run_healing_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic Self-Healing — Databricks Schema Drift API",
    description=(
        "Webhook receiver that triggers a LangGraph multi-agent workflow to "
        "automatically detect, diagnose, test, and fix Databricks Delta Lake "
        "schema drift failures."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class JobFailureEvent(BaseModel):
    """Payload sent by Databricks (or the Airflow failure callback) when a job
    fails due to a schema drift error.

    Attributes
    ----------
    job_id:
        Databricks job identifier.
    run_id:
        Specific run identifier for the failed execution (used for audit logs).
    event_type:
        Should be ``"job_failure"`` for schema drift events.
    error:
        Raw error / log text from the failed run.
    """

    job_id: str = Field(..., description="Databricks job ID that failed.")
    run_id: str = Field(..., description="Databricks run ID of the failed execution.")
    event_type: str = Field(
        "job_failure",
        description="Event type; must be 'job_failure' to trigger healing.",
    )
    error: str = Field(..., description="Raw error logs from the failed job run.")


class WebhookResponse(BaseModel):
    """Immediate acknowledgement returned to the caller."""

    message: str
    job_id: str


# ---------------------------------------------------------------------------
# Background task wrapper
# ---------------------------------------------------------------------------

def _run_workflow_sync(job_id: str, error_logs: str) -> None:
    """Run the async healing workflow in a background thread via asyncio."""
    try:
        asyncio.run(run_healing_workflow(job_id=job_id, error_logs=error_logs))
    except Exception:
        logger.exception("Self-healing workflow failed for job %s", job_id)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/webhook/job-failure",
    response_model=WebhookResponse,
    summary="Receive a Databricks job failure event",
    response_description="Acknowledgement that the healing workflow was queued.",
)
async def job_failure_webhook(
    event: JobFailureEvent,
    background_tasks: BackgroundTasks,
) -> WebhookResponse:
    """Trigger the self-healing workflow when a Databricks job fails.

    The workflow runs asynchronously in the background so this endpoint returns
    immediately (HTTP 202-style), allowing Databricks to continue without
    waiting for the healing process to complete.

    Parameters
    ----------
    event:
        JSON body describing the failed job.
    background_tasks:
        FastAPI dependency used to schedule the healing coroutine.

    Raises
    ------
    HTTPException
        422 if the request body does not match the ``JobFailureEvent`` schema.
    """
    if event.event_type != "job_failure":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported event_type '{event.event_type}'. Expected 'job_failure'.",
        )

    logger.info(
        "Received failure event for job_id=%s run_id=%s",
        event.job_id,
        event.run_id,
    )

    background_tasks.add_task(_run_workflow_sync, event.job_id, event.error)

    return WebhookResponse(
        message="Self-healing workflow queued successfully.",
        job_id=event.job_id,
    )


@app.get("/health", summary="Liveness probe")
async def health_check() -> dict:
    """Return a simple ``{"status": "ok"}`` for health checks."""
    return {"status": "ok"}
