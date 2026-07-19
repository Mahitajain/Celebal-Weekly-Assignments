"""
Orchestrator Agent, built as a LangGraph `StateGraph`.

Graph shape:

                    ┌───────────┐
        ┌──────────►│  sql_node │──────────┐
        │           └───────────┘          │
  ┌───────────┐                     ┌──────▼──────┐
  │ classify  │────────────────────►│  synthesize │──► END
  └───────────┘                     └──────▲──────┘
        │           ┌───────────┐          │
        └──────────►│  rag_node │──────────┘
        │           └───────────┘
        │           (hybrid runs both sql_node and rag_node)
        │
        └──────────► clarify_node ──► END
        └──────────► unsupported_node ──► END

`classify` calls the Orchestrator's routing LLM (see `classifier.py`) and
sets `state["route"]`. A conditional edge then sends state to exactly the
node(s) that route implies. `hybrid` fans out to both agents; `synthesize`
merges whatever agent results are present into one unified, cited answer.
Falling back to a plain Python function if `langgraph` isn't installed
would duplicate this exact control flow with less clarity, so this graph
*is* the orchestration logic, not a wrapper around it.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.llm.client import LLMClient, get_llm_client
from app.orchestrator.classifier import classify
from app.orchestrator.state import AgentTrace, OrchestratorState
from app.rag.agent import RAGAgent
from app.rag.retriever import Retriever
from app.sql_agent.agent import SQLAgent
from app.utils.logging_config import get_logger
from sqlalchemy.engine import Engine

logger = get_logger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You merge results from a hospital database query and/or hospital policy \
documents into one short, coherent answer for hospital staff. Do not invent information beyond what is \
given. If both a data result and a policy result are present, connect them clearly (e.g. "3 patients \
require pre-op clearance; per policy, clearance requires ..."). Keep it to a few sentences."""


class Orchestrator:
    def __init__(self, engine: Engine, retriever: Retriever, llm: LLMClient | None = None):
        self.llm = llm or get_llm_client()
        self.sql_agent = SQLAgent(engine, llm=self.llm)
        self.rag_agent = RAGAgent(retriever, llm=self.llm)
        self.graph = self._build_graph()

    # -- Nodes -------------------------------------------------------------
    def _classify_node(self, state: OrchestratorState) -> OrchestratorState:
        decision = classify(state["question"], state.get("history", []), self.llm)
        logger.info("Routed question to '%s' (confidence=%.2f): %s", decision.route, decision.confidence, decision.reasoning)
        return {
            "route": decision.route,
            "route_confidence": decision.confidence,
            "route_reasoning": decision.reasoning,
            "trace": [AgentTrace(agent="orchestrator", success=True, detail=f"routed:{decision.route}")],
        }

    def _sql_node(self, state: OrchestratorState) -> OrchestratorState:
        history_text = "\n".join(f"{t['role']}: {t['content']}" for t in state.get("history", []))
        result = self.sql_agent.answer(state["question"], history=history_text)
        return {
            "sql_result": {
                "success": result.success,
                "sql": result.sql,
                "answer": result.answer,
                "explanation": result.explanation,
                "columns": result.execution.columns if result.execution else [],
                "rows": result.execution.rows if result.execution else [],
                "row_count": result.execution.row_count if result.execution else 0,
                "used_llm": result.used_llm,
                "error": result.error,
            },
            "trace": [AgentTrace(agent="sql_agent", success=result.success, detail=result.error or "ok")],
        }

    def _rag_node(self, state: OrchestratorState) -> OrchestratorState:
        result = self.rag_agent.answer(state["question"])
        return {
            "rag_result": {
                "success": result.success,
                "answer": result.answer,
                "citations": [
                    {
                        "document": c.doc_title,
                        "section_number": c.section_number,
                        "section_title": c.section_title,
                        "score": round(c.score, 3),
                    }
                    for c in result.citations
                ],
                "used_llm": result.used_llm,
                "error": result.error,
            },
            "trace": [AgentTrace(agent="rag_agent", success=result.success, detail=result.error or "ok")],
        }

    def _synthesize_node(self, state: OrchestratorState) -> OrchestratorState:
        sql_result = state.get("sql_result")
        rag_result = state.get("rag_result")

        if sql_result and rag_result:
            if self.llm.available and (sql_result["success"] or rag_result["success"]):
                context = (
                    f"DATABASE RESULT: {sql_result['answer'] if sql_result['success'] else sql_result['error']}\n\n"
                    f"POLICY RESULT: {rag_result['answer'] if rag_result['success'] else rag_result['error']}"
                )
                response = self.llm.complete(
                    system=SYNTHESIS_SYSTEM_PROMPT,
                    user=f"Question: {state['question']}\n\n{context}",
                    temperature=0.1,
                    max_tokens=400,
                )
                final = response.text.strip() if response else context
            else:
                parts = []
                if sql_result["success"]:
                    parts.append(f"Data: {sql_result['answer']}")
                if rag_result["success"]:
                    parts.append(f"Policy: {rag_result['answer']}")
                final = "\n\n".join(parts) or "Neither the database nor the policy documents could answer this."
        elif sql_result:
            final = sql_result["answer"] if sql_result["success"] else (sql_result["error"] or "I couldn't answer that from the database.")
        elif rag_result:
            final = rag_result["answer"] if rag_result["success"] else (rag_result["error"] or "I couldn't find that in the policy documents.")
        else:
            final = "I wasn't able to process this question."

        return {"final_answer": final}

    def _clarify_node(self, state: OrchestratorState) -> OrchestratorState:
        return {
            "final_answer": (
                "I'm not sure whether that's a question about patient/admissions data or about hospital "
                "policy -- could you rephrase with a bit more detail? For example, 'How many diabetic "
                "patients are over 60?' (data) or 'What is the visitor policy for the ICU?' (policy)."
            ),
            "clarification_prompt": state["question"],
        }

    def _unsupported_node(self, state: OrchestratorState) -> OrchestratorState:
        return {
            "final_answer": (
                "That doesn't look like a question about hospital patient data or hospital policy, which "
                "are the two things I can help with here."
            )
        }

    # -- Graph construction --------------------------------------------------
    def _route_selector(self, state: OrchestratorState) -> list[str]:
        route = state["route"]
        return {
            "sql": ["sql_node"],
            "rag": ["rag_node"],
            "hybrid": ["sql_node", "rag_node"],
            "clarify": ["clarify_node"],
            "unsupported": ["unsupported_node"],
        }[route]

    def _build_graph(self):
        graph = StateGraph(OrchestratorState)
        graph.add_node("classify", self._classify_node)
        graph.add_node("sql_node", self._sql_node)
        graph.add_node("rag_node", self._rag_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("clarify_node", self._clarify_node)
        graph.add_node("unsupported_node", self._unsupported_node)

        graph.set_entry_point("classify")
        graph.add_conditional_edges("classify", self._route_selector, ["sql_node", "rag_node", "clarify_node", "unsupported_node"])
        graph.add_edge("sql_node", "synthesize")
        graph.add_edge("rag_node", "synthesize")
        graph.add_edge("synthesize", END)
        graph.add_edge("clarify_node", END)
        graph.add_edge("unsupported_node", END)
        return graph.compile()

    # -- Public entry point ----------------------------------------------------
    def run(self, question: str, history: list | None = None) -> OrchestratorState:
        initial: OrchestratorState = {"question": question, "history": history or [], "trace": []}
        return self.graph.invoke(initial)
