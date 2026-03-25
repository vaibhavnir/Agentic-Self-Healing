import uuid

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from agent_workflow import app_graph

app = FastAPI(title="Agentic Triage Server")


# Define the expected webhook payload from Airflow
class AirflowAlert(BaseModel):
    dag_id: str
    task_id: str
    error_message: str


async def run_healing_workflow(dag_id: str, error_message: str):
    """Executes the LangGraph workflow in an isolated thread."""

    # 1. Thread Isolation: Create a unique ID for this specific failure
    incident_id = f"{dag_id}_{uuid.uuid4().hex[:6]}"
    thread_config = {"configurable": {"thread_id": incident_id}}

    initial_state = {
        "dag_id": dag_id,
        "error_logs": error_message,
    }

    print(f"\n--- Initiating Incident Resolution: {incident_id} ---")

    # 2. Execute the graph asynchronously, streaming agent progress
    async for output in app_graph.astream(initial_state, config=thread_config):
        for node_name, node_output in output.items():
            print(f"  [{incident_id}] Node '{node_name}' output: {node_output}")

    print(f"--- Incident {incident_id} Resolved ---\n")


@app.post("/webhook/airflow-alert")
async def handle_alert(alert: AirflowAlert, bg_tasks: BackgroundTasks):
    """Airflow hits this endpoint when a DAG fails."""

    # Hand the heavy lifting to a background task so Airflow isn't left hanging
    bg_tasks.add_task(run_healing_workflow, alert.dag_id, alert.error_message)

    return {
        "status": "Agents Dispatched",
        "dag_id": alert.dag_id,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
