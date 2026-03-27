# Agentic Self-Healing ETL Pipeline

A **LangGraph-based Agentic AI workflow** that autonomously detects, diagnoses, and recovers from ETL pipeline failures — without hard-coded rules.

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

---

## Project Structure

```
.
├── config/
│   └── healing_strategies.yaml     # Strategy catalogue and parameters
├── src/
│   ├── __init__.py
│   ├── state.py                    # HealingState TypedDict
│   ├── strategies.py               # RecoveryStrategy ABC + 6 concrete strategies
│   ├── adaptive_healer.py          # Learning confidence scorer
│   ├── llm_interface.py            # LLM wrapper (real + rule-based fallback)
│   ├── nodes.py                    # LangGraph node functions
│   ├── workflow.py                 # StateGraph definition & compilation
│   └── config_loader.py            # YAML config loader
├── tests/
│   ├── test_strategies.py          # Unit tests for all strategies
│   ├── test_adaptive_healer.py     # Unit tests for AdaptiveHealer
│   └── test_workflow.py            # End-to-end integration tests
├── main.py                         # Entry point with demo scenarios
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Configure an LLM

```bash
export OPENAI_API_KEY=sk-...
```

Without an API key the system uses a config-driven rule-based fallback — all features still work.

### 3. Run the demo

```bash
python main.py
```

Sample output:

```
=================================================================
  Scenario 1: ✅ RECOVERED
=================================================================
  Error type   : timeout
  Diagnosis    : ETL pipeline failure: Query timed out after 30s
  Strategy     : retry
  Confidence   : 0.54
  Attempts     : 1
  Result msg   : Retry #1 scheduled after 1.00s back-off

  Reasoning trace:
    [detect] Error signal received from 'postgres_loader': type='timeout'...
    [diagnose] Issue: 'ETL pipeline failure: ...' | error_type='timeout'
    [plan_recovery] Selected 'retry' (confidence=0.54). Reasoning: ...
    [execute] Strategy 'retry' → success=True: Retry #1 scheduled ...
    [validate] Recovery SUCCESSFUL via 'retry'. Pipeline returning to healthy state.
```

### 4. Use as a library

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

  - name: "circuit_breaker"
    applicable_errors: ["service_unavailable"]
    parameters:
      failure_threshold: 5
      recovery_timeout_s: 300

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

All 32 tests (unit + integration) run without an LLM API key.

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Enables GPT-4o-mini for diagnosis and strategy selection |
| `HEALING_CONFIG` | Path to an alternative `healing_strategies.yaml` |
