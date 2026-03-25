import asyncio
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# 1. Define the Shared Memory (State)
class PipelineState(TypedDict):
    dag_id: str
    error_logs: str
    root_cause: str
    proposed_code_fix: str
    qa_passed: bool
    qa_feedback: str
    status_message: str


# 2. Define the Agents (Nodes)
async def detective_agent(state: PipelineState):
    print(f"[{state['dag_id']}] 🕵️ Detective: Analyzing error logs...")
    # Mock LLM/RAG call
    await asyncio.sleep(1)
    return {"root_cause": "Schema mismatch detected in source database."}


async def engineer_agent(state: PipelineState):
    print(f"[{state['dag_id']}] 🛠️ Engineer: Writing dbt SQL fix...")
    # Mock LLM/MCP tool call to write code
    await asyncio.sleep(1)

    # Simulate an iterative fix based on QA feedback
    if state.get("qa_feedback"):
        fix = "SELECT Acct_Mgr_Name AS Account_Manager FROM source"
    else:
        fix = "SELECT Acct_Mgr_Name FROM source"  # First draft (buggy)

    return {"proposed_code_fix": fix}


async def qa_agent(state: PipelineState):
    print(f"[{state['dag_id']}] 🧪 QA: Running dbt tests...")
    await asyncio.sleep(1)

    # Mock testing logic: It fails the first time, passes the second
    if "Account_Manager" in state.get("proposed_code_fix", ""):
        return {"qa_passed": True, "qa_feedback": "Tests passed."}
    else:
        return {"qa_passed": False, "qa_feedback": "Missing alias 'Account_Manager'."}


async def deployment_agent(state: PipelineState):
    print(f"[{state['dag_id']}] 🚀 Deploy: Merging PR & Alerting Slack...")
    await asyncio.sleep(1)
    return {"status_message": f"Fixed and deployed for {state['dag_id']}"}


# 3. Routing Logic (The A2A Loop)
def route_qa(state: PipelineState):
    if state.get("qa_passed"):
        return "deploy"
    return "rewrite"


# 4. Build and Compile the Graph
def build_graph():
    workflow = StateGraph(PipelineState)

    workflow.add_node("Detective", detective_agent)
    workflow.add_node("Engineer", engineer_agent)
    workflow.add_node("QA", qa_agent)
    workflow.add_node("Deploy", deployment_agent)

    workflow.add_edge(START, "Detective")
    workflow.add_edge("Detective", "Engineer")
    workflow.add_edge("Engineer", "QA")

    # The Cyclic Edge for self-healing rewrites
    workflow.add_conditional_edges(
        "QA",
        route_qa,
        {"deploy": "Deploy", "rewrite": "Engineer"}
    )
    workflow.add_edge("Deploy", END)

    # Add memory checkpointer for State Isolation (Multithreading)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


app_graph = build_graph()
