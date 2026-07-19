"""Shared state passed through the orchestrator graph.

Kept as a plain TypedDict (rather than a class with methods) because
LangGraph's `StateGraph` operates on dict-like state objects that each
node reads from and returns partial updates to -- this is the same shape
whether the graph is actually run by LangGraph or by the fallback
sequential runner in `graph.py`.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

Route = Literal["sql", "rag", "hybrid", "clarify", "unsupported"]


class ConversationTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class AgentTrace(TypedDict, total=False):
    agent: str
    success: bool
    detail: str


class OrchestratorState(TypedDict, total=False):
    question: str
    history: list[ConversationTurn]
    route: Route
    route_confidence: float
    route_reasoning: str
    sql_result: dict | None
    rag_result: dict | None
    trace: Annotated[list[AgentTrace], operator.add]
    final_answer: str
    clarification_prompt: str | None
