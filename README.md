# Agentic Self-Healing ETL Pipeline

A **LangGraph-based Agentic AI workflow** that autonomously detects, diagnoses, and recovers from ETL pipeline failures — without hard-coded rules.

Integrated with **Databricks Jobs** for re-triggering failed job runs and exposed as a **FastAPI** service that handles multiple failure instances concurrently with full thread safety.

---

## Architecture

```
[detect] → [diagnose] → [plan_recovery] → [execute] → [validate]
                                                            │
                              ┌─────────────────────────────┘
                              ↓
                    confidence ≥ threshold
                    AND attempts < max    →  [execute]  (retry loop)
                              ↓
                    !recovered AND attempts exhausted
                              →  [escalate] → END
                              ↓
                    recovered → END
```

Every node reads from and writes to a shared `HealingState` TypedDict; LangGraph merges partial updates automatically.

### FastAPI layer

```
Databricks job / ETL pipeline
        │
        │  POST /heal or POST /heal/batch
        ▼
  ┌───────────────────────────────────────┐
  │           FastAPI Service             │
  │  ┌────────────────────────────────┐   │
  │  │  /heal/batch (asyncio.gather)  │   │
  │  │  ├─ workflow.ainvoke(failure1) │   │
  │  │  ├─ workflow.ainvoke(failure2) │   │   ← concurrent, thread-safe
  │  │  └─ workflow.ainvoke(failureN) │   │
  │  └────────────────────────────────┘   │
  │  Shared: AdaptiveHealer (Lock)        │
  │          compiled LangGraph workflow  │
  └───────────────────────────────────────┘
        │
        │  DatabricksJobsClient (httpx async)
        ▼
  Databricks Jobs REST API
```

---

## Key Design Principles

| Principle | Implementation |
|---|---|
| **No hard-coded rules** | LLM diagnosis + config-driven strategy matching |
| **Self-learning** | `AdaptiveHealer` tracks per-(error_type, action) success rates and adjusts confidence |
| **Configurable** | All strategy parameters live in `config/healing_strategies.yaml` |
| **Extensible** | Add a new strategy by subclassing `RecoveryStrategy` and adding an entry to the YAML |
| **Graceful degradation** | Falls back to highest-confidence strategy when no LLM key is present; escalates to humans when confidence is too low |
| **Auditable** | Every decision is appended to `reasoning_trace` in the state |
| **Thread-safe** | `threading.Lock` guards `AdaptiveHealer` history and `CircuitBreakerStrategy` class-level state |
| **Parallel** | `POST /heal/batch` dispatches all failures concurrently via `asyncio.gather` |

---

## Project Structure

```
.
├── api/
│   ├── __init__.py
│   ├── app.py                      # FastAPI application, routes, lifespan
│   ├── dependencies.py             # Singleton workflow + healer (AppState)
│   └── models.py                   # Pydantic request/response models
├── config/
│   └── healing_strategies.yaml     # Strategy catalogue and parameters
├── src/
│   ├── __init__.py
│   ├── state.py                    # HealingState TypedDict (+ Databricks fields)
│   ├── strategies.py               # RecoveryStrategy ABC + 7 concrete strategies
│   ├── adaptive_healer.py          # Thread-safe learning confidence scorer
│   ├── databricks_client.py        # Async Databricks Jobs REST API client
│   ├── llm_interface.py            # LLM wrapper (real + rule-based fallback)
│   ├── nodes.py                    # LangGraph node functions
│   ├── workflow.py                 # StateGraph definition & compilation
│   └── config_loader.py            # YAML config loader
├── tests/
│   ├── test_strategies.py          # Unit tests for all strategies
│   ├── test_adaptive_healer.py     # Unit tests for AdaptiveHealer
│   ├── test_workflow.py            # End-to-end integration tests
│   ├── test_api.py                 # FastAPI endpoint tests
│   └── test_thread_safety.py       # Concurrency / thread-safety tests
├── main.py                         # CLI entry point with demo scenarios
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Configure integrations

```bash
# LLM-powered diagnosis (falls back to config-driven rules when absent)
export OPENAI_API_KEY=sk-...

# Databricks job re-triggering
export DATABRICKS_HOST=https://adb-1234567890.azuredatabricks.net
export DATABRICKS_TOKEN=dapi...

# Override config file location
export HEALING_CONFIG=path/to/healing_strategies.yaml
```

### 3. Run the CLI demo

```bash
python main.py
```

### 4. Start the FastAPI server

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Interactive API docs available at `http://localhost:8000/docs`.

### 5. Use as a library

```python
import asyncio
from src.adaptive_healer import AdaptiveHealer
from src.config_loader import load_config
from src.state import HealingState
from src.workflow import create_self_healing_workflow

config = load_config()
healer = AdaptiveHealer(outcomes_file="data/healing_outcomes.json")
workflow = create_self_healing_workflow(
    strategy_config=config["strategies"],
    adaptive_healer=healer,
)

error_signal = {
    "type": "timeout",
    "source": "my_etl_job",
    "message": "Connection timed out",
    "last_checkpoint": "cp_2024_01_01",
}

initial_state: HealingState = {
    "error_signal": error_signal,
    "diagnosed_issue": "", "error_type": "",
    "available_actions": [], "selected_action": "",
    "action_parameters": {}, "execution_result": {},
    "confidence_score": 0.0, "attempt_count": 0,
    "is_recovered": False, "reasoning_trace": [],
}

final_state = asyncio.run(workflow.ainvoke(initial_state))
print("Recovered:", final_state["is_recovered"])
print("Trace:", final_state["reasoning_trace"])
```

---

## FastAPI Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{"status": "ok", "workflow_ready": true}` |
| `POST` | `/heal` | Heal a **single** ETL failure |
| `POST` | `/heal/batch` | Heal **multiple** failures **in parallel** (thread-safe) |
| `GET` | `/outcomes` | Adaptive healer learning summary, sorted by success rate |

### POST /heal

```bash
curl -X POST http://localhost:8000/heal \
  -H "Content-Type: application/json" \
  -d '{
    "error_signal": {
      "type": "timeout",
      "source": "postgres_loader",
      "message": "Query timed out after 30s",
      "last_checkpoint": "cp_001"
    }
  }'
```

### POST /heal/batch — parallel processing

```bash
curl -X POST http://localhost:8000/heal/batch \
  -H "Content-Type: application/json" \
  -d '{
    "failures": [
      {"type": "timeout",            "source": "job_a", "message": "timeout",    "last_checkpoint": "cp1"},
      {"type": "data_quality_issue", "source": "job_b", "message": "bad data",   "null_pct": 20, "total_records": 1000, "last_checkpoint": "cp2"},
      {"type": "memory_exceeded",    "source": "job_c", "message": "OOM",        "last_checkpoint": "cp3"}
    ]
  }'
```

All three failures are dispatched simultaneously via `asyncio.gather`. Total latency is bounded by the slowest individual failure, not their sum.

---

## Databricks Integration

### How it works

1. When a Databricks job fails, it calls `POST /heal` (or `/heal/batch`) with the error signal.
2. The workflow diagnoses the failure and selects the best recovery strategy.
3. If the `databricks_retry` strategy is selected, the `DatabricksJobsClient` re-triggers the job via the [Databricks Jobs REST API v2.1](https://docs.databricks.com/api/workspace/jobs).

### Error signal for Databricks failures

Include `databricks_job_id` in the error signal to enable job re-triggering:

```json
{
  "type": "databricks_job_failed",
  "source": "my_etl_notebook",
  "message": "Job run 12345 failed with exit code 1",
  "databricks_job_id": "987654321",
  "databricks_notebook_params": {"env": "prod", "date": "2024-01-15"},
  "last_checkpoint": "cp_databricks_001"
}
```

### Calling from a Databricks notebook

```python
import requests

response = requests.post(
    "http://your-healing-service:8000/heal",
    json={
        "error_signal": {
            "type": "databricks_job_failed",
            "source": dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get(),
            "message": str(exception),
            "databricks_job_id": spark.conf.get("spark.databricks.job.id", ""),
            "last_checkpoint": "cp_" + str(current_date),
        }
    },
    timeout=30,
)
result = response.json()
if result["is_recovered"]:
    print(f"Recovery succeeded via: {result['selected_action']}")
else:
    raise RuntimeError("Escalated to human operators")
```

### DatabricksJobsClient (direct usage)

```python
from src.databricks_client import DatabricksJobsClient

client = DatabricksJobsClient.from_env()  # reads DATABRICKS_HOST + DATABRICKS_TOKEN

run_id = await client.trigger_run(job_id="123456", notebook_params={"date": "2024-01-15"})
status  = await client.get_run_status(run_id)
print(status["life_cycle_state"])  # RUNNING / TERMINATED / ...
```

---

## Thread Safety

All shared mutable state is protected:

| Component | Shared state | Protection |
|---|---|---|
| `AdaptiveHealer` | `_history` dict + JSON file | `threading.Lock` on all reads, writes, and persistence |
| `CircuitBreakerStrategy` | class-level `_failure_counts` / `_open_since` | `threading.Lock` wrapping every mutation |
| `Compiled LangGraph workflow` | **stateless** — each `ainvoke` gets its own `HealingState` | No lock needed |

The FastAPI `/heal/batch` endpoint safely dispatches dozens of concurrent `ainvoke` calls, with all outcome recordings serialised through the adaptive healer's lock.

---

## Configuration

Edit `config/healing_strategies.yaml` to add or tune strategies without touching Python code:

```yaml
strategies:
  - name: "retry"
    applicable_errors: ["timeout", "connection_refused"]
    parameters:
      max_retries: 3
      backoff_multiplier: 2
      initial_delay_ms: 1000

  - name: "databricks_retry"
    applicable_errors: ["databricks_job_failed", "databricks_timeout"]
    parameters:
      max_retries: 2
      poll_interval_s: 30

escalation:
  confidence_threshold: 0.3   # escalate to humans below this

feedback:
  persist_outcomes: true
  outcomes_file: "data/healing_outcomes.json"
  min_samples_for_confidence: 5
```

---

## Built-in Recovery Strategies

| Strategy | Applicable Errors | Description |
|---|---|---|
| `retry` | timeout, connection_refused, transient_error | Exponential back-off retry |
| `circuit_breaker` | service_unavailable, rate_limit_exceeded | Open/half-open circuit management |
| `rollback` | data_corruption, schema_mismatch, write_failure | Restore last checkpoint |
| `data_validation` | data_quality_issue, null_values_exceeded, type_mismatch | Quarantine bad records |
| `resource_scaling` | memory_exceeded, cpu_throttled, disk_full | Request more compute resources |
| `dead_letter_queue` | unrecoverable_error, max_retries_exceeded | Enqueue for manual review |
| `databricks_retry` | databricks_job_failed, databricks_timeout | Re-trigger Databricks job via REST API |

---

## Extending the System

### Add a new strategy

1. Subclass `RecoveryStrategy` in `src/strategies.py`:

```python
class MyNewStrategy(RecoveryStrategy):
    name = "my_new_strategy"

    def can_handle(self, error_context: dict) -> float:
        return 0.9 if error_context.get("type") == "my_error" else 0.0

    async def execute(self, context: dict) -> dict:
        # recovery logic here
        return {"success": True, "message": "Fixed!", "metadata": {}}
```

2. Register it in `build_strategy_registry` (the `_class_map` dict in `strategies.py`).

3. Add a YAML entry in `config/healing_strategies.yaml` — no other code changes needed.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

All 50 tests (unit + integration + API + thread-safety) run without an LLM API key or Databricks credentials.

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Enables GPT-4o-mini for diagnosis and strategy selection |
| `HEALING_CONFIG` | Path to an alternative `healing_strategies.yaml` |
| `DATABRICKS_HOST` | Databricks workspace URL (e.g. `https://adb-1234567890.azuredatabricks.net`) |
| `DATABRICKS_TOKEN` | Databricks personal access token or OAuth M2M token |
