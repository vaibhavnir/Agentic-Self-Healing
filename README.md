# Agentic Self-Healing — Databricks Schema Drift Pipeline

A **LangGraph-based multi-agent AI workflow** that automatically detects, diagnoses, tests, and fixes **Databricks Delta Lake schema drift** failures — with no human intervention required.

---

## Overview

In Databricks, **schema drift** occurs when a source system (an operational database, a raw JSON stream, etc.) introduces a new column, changes a data type, or drops a field. When your scheduled PySpark job tries to write mutated data into a strict Delta table, it throws:

```
AnalysisException: A schema mismatch detected when writing to the Delta table…
```

This project assembles a **four-agent LangGraph team** that self-heals the pipeline end-to-end:

```
START → Detective → Engineer → QA ──(pass)──→ Deploy → END
                       ↑            │
                       └──(fail)────┘
```

---

## Agent Roles

| Agent | Role | Key Action |
|---|---|---|
| 🕵️ **Detective** | Schema Analyzer | Queries Unity Catalog (`DESCRIBE TABLE`) and compares it to the incoming DataFrame schema to identify the exact drift |
| 🛠️ **Engineer** | PySpark / SQL Coder | Decides the fix strategy: `mergeSchema` option (first attempt) or explicit `ALTER TABLE` DDL (if QA rejects) |
| 🧪 **QA** | Sandbox Tester | Creates a Zero-Copy Clone (`CLONE`) of the production table and validates the fix in isolation, enforcing governance rules |
| 🚀 **Deploy** | Databricks Jobs Manager | Applies the approved DDL to production via Unity Catalog REST API and restarts the failed Databricks job |

---

## Why This Matters

### 1 — Zero-Copy Cloning for Safe QA
Databricks can clone terabytes of data using metadata pointers (`CREATE TABLE … CLONE …`) in milliseconds. The QA agent tests schema evolution on a perfect replica of production — if anything goes wrong, the main Delta tables are never touched.

### 2 — Governance-Aware Healing
The QA agent acts as a guardrail. Enterprise environments typically forbid implicit `mergeSchema=True` because upstream bugs could inject garbage columns into a clean data warehouse. The QA agent enforces this policy by rejecting lazy fixes and requiring the Engineer to write explicit `ALTER TABLE` DDL.

### 3 — Unity Catalog Integration
Because Unity Catalog tracks lineage centrally, the Detective Agent can query which downstream dashboards depend on the drifting table and alert specific stakeholders before applying any changes.

### 4 — Retry Loop
If QA rejects the Engineer's first proposal, the graph automatically routes back to the Engineer for a revised fix — no manual re-run needed.

---

## Repository Structure

```
Agentic-Self-Healing/
├── agent_workflow.py   # LangGraph agent graph (Detective → Engineer → QA → Deploy)
├── api.py              # FastAPI webhook server — receives Databricks failure events
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Access to a Databricks workspace (Unity Catalog enabled)
- Databricks personal access token with job and SQL Warehouse permissions

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the local smoke-test

```bash
python agent_workflow.py
```

This simulates a full healing cycle for a sample `AnalysisException` log, printing the output of each agent.

### Start the FastAPI webhook server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs are available at `http://localhost:8000/docs`.

### Configure Databricks notifications

1. In the Databricks UI, open your job → **Edit** → **Notifications**.
2. Add a **Webhook** destination pointing to `https://<your-host>/webhook/job-failure`.
3. Databricks will POST a JSON payload like this on every failure:

```json
{
  "job_id": "98765",
  "run_id": "112233",
  "event_type": "job_failure",
  "error": "AnalysisException: A schema mismatch detected..."
}
```

The webhook returns immediately (the healing workflow runs in the background) so Databricks is never kept waiting.

---

## How the Healing Workflow Operates

### First pass (mergeSchema)

```
Detective detects: new column 'loyalty_tier' (StringType)
Engineer proposes: df.write … .option("mergeSchema", "true")
QA rejects:        governance policy forbids implicit schema evolution
```

### Second pass (ALTER TABLE)

```
Engineer re-proposes: ALTER TABLE main.sales.customers ADD COLUMN loyalty_tier STRING;
QA approves:          fix validated on Zero-Copy Clone
Deploy applies:       DDL executed in production; job restarted via Jobs API
```

---

## Production Integration Notes

The agent nodes contain clearly marked `# --- Production replacement ---` blocks showing exactly where to plug in:

- **Databricks SQL Warehouse** calls via `databricks-sdk` or an MCP tool
- **Zero-Copy Clone** creation and teardown
- **Databricks Jobs REST API** (`POST /api/2.1/jobs/run-now`)
- **Databricks Git Folders (Repos)** for committing updated PySpark notebooks

---

## Tech Stack

| Technology | Purpose |
|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful multi-agent graph orchestration |
| [FastAPI](https://fastapi.tiangolo.com/) | Webhook HTTP server |
| [Databricks SDK](https://docs.databricks.com/dev-tools/sdk-python.html) | Unity Catalog, Jobs API, SQL Warehouse |
| [Delta Lake](https://delta.io/) | ACID-compliant table format on Databricks |
| [PySpark](https://spark.apache.org/docs/latest/api/python/) | Distributed data processing |

