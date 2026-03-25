import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


# 1. The Webhook Callback
def trigger_agentic_triage(context):
    """Extracts failure info and alerts FastAPI."""
    task_instance = context.get("task_instance")

    payload = {
        "dag_id": task_instance.dag_id,
        "task_id": task_instance.task_id,
        "error_message": str(context.get("exception")),
    }

    api_url = "http://localhost:8000/webhook/airflow-alert"

    try:
        # Send the alert to FastAPI
        requests.post(api_url, json=payload, timeout=5)
        print(f"Alert sent to AI Triage for {task_instance.dag_id}.")
    except Exception as e:
        print(f"Failed to trigger AI workflow: {e}")


# 2. Attach to Default Args
default_args = {
    "owner": "data_team",
    "start_date": datetime(2024, 1, 1),
    "retries": 0,
    "on_failure_callback": trigger_agentic_triage,  # Trigger on ANY task failure
}


# 3. Simulate a Failing DAG
with DAG(
    "salesforce_ingestion_pipeline",
    default_args=default_args,
    schedule_interval="@daily",
) as dag:

    def simulate_crash():
        raise ValueError("FATAL: Column 'Account_Manager' missing in source data.")

    run_dbt_models = PythonOperator(
        task_id="run_dbt_transformations",
        python_callable=simulate_crash,
    )
