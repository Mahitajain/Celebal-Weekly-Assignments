from __future__ import annotations

from app.orchestrator.graph import Orchestrator


def test_routes_data_question_to_sql(db_engine, retriever):
    orch = Orchestrator(db_engine, retriever)
    state = orch.run("How many diabetic patients are there?")
    assert state["route"] == "sql"
    assert state["sql_result"] is not None
    assert state.get("rag_result") is None
    assert state["final_answer"]


def test_routes_policy_question_to_rag(db_engine, retriever):
    orch = Orchestrator(db_engine, retriever)
    state = orch.run("What is the visitor policy for the ICU?")
    assert state["route"] == "rag"
    assert state["rag_result"] is not None
    assert state.get("sql_result") is None
    assert state["final_answer"]


def test_routes_mixed_question_to_hybrid(db_engine, retriever):
    orch = Orchestrator(db_engine, retriever)
    state = orch.run(
        "What is the average length of stay for cancer patients and what does the discharge "
        "policy say about long stays?"
    )
    assert state["route"] == "hybrid"
    assert state["sql_result"] is not None
    assert state["rag_result"] is not None


def test_unsupported_question_gets_graceful_response(db_engine, retriever):
    orch = Orchestrator(db_engine, retriever)
    state = orch.run("What is the meaning of life?")
    assert state["route"] in ("clarify", "unsupported")
    assert state["final_answer"]


def test_trace_records_every_agent_invoked(db_engine, retriever):
    orch = Orchestrator(db_engine, retriever)
    state = orch.run("How many emergency admissions are there?")
    agents_invoked = {t["agent"] for t in state["trace"]}
    assert "orchestrator" in agents_invoked
    assert "sql_agent" in agents_invoked
