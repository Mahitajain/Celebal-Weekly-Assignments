"""
Intent classification for the Orchestrator Agent.

"The routing should be intelligent -- not simple keyword matching" is the
brief, so the primary path here is an LLM classification call that reads
the actual question (and recent conversation history, so a follow-up like
"and what about elective ones?" inherits context) and returns a
structured decision.

A keyword-based heuristic (`_heuristic_route`) exists only as a fallback
for when no LLM is configured, so the project still runs end-to-end
without an API key. It is intentionally kept separate from, and simpler
than, the LLM path -- it is not presented as "the routing logic," it is
the degraded-mode safety net.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.llm.client import LLMClient
from app.orchestrator.state import ConversationTurn, Route

CLASSIFIER_SYSTEM_PROMPT = """You are the routing brain of a hospital assistant. Classify the user's \
question into exactly one route:

- "sql": the question asks about structured patient/admission data -- counts, filters, averages, \
lists of patients, doctors, admission dates, billing, test results, etc. Answerable by querying a \
patient-admissions database.
- "rag": the question asks about hospital policy, procedures, or documented rules -- visitor hours, \
discharge procedures, infection control, pre-surgery requirements, privacy, billing policy (the \
POLICY, not a specific patient's bill). Answerable from hospital policy documents.
- "hybrid": the question genuinely needs BOTH structured data AND policy context to answer fully \
(e.g. "Which of Dr. Sharma's patients need pre-surgery clearance and what's required?").
- "clarify": the question is too ambiguous to route confidently (e.g. it could plausibly be either, \
or lacks the detail needed to answer either way).
- "unsupported": the question is unrelated to hospital data or hospital policy entirely.

Respond ONLY with a JSON object, no markdown fences, no commentary:
{{"route": "sql|rag|hybrid|clarify|unsupported", "confidence": 0.0-1.0, "reasoning": "one short sentence"}}
"""

CLASSIFIER_USER_TEMPLATE = """Conversation so far (most recent last, may be empty):
{history}

Current question: {question}"""

_SQL_HINTS = (
    "how many", "count", "average", "list", "show", "patients", "admitted", "admission",
    "doctor", "billing", "insurance", "age", "room", "medication", "test result",
    "diabetic", "diabetes", "cancer", "asthma", "obesity", "hypertension", "arthritis",
)
_RAG_HINTS = (
    "policy", "procedure", "protocol", "visitor", "visiting hours", "discharge process",
    "infection control", "consent", "privacy", "confidentiality", "pre-surgery", "before surgery",
    "required documents", "hipaa", "financial assistance",
)


@dataclass
class RoutingDecision:
    route: Route
    confidence: float
    reasoning: str


def _format_history(history: list[ConversationTurn]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)


def _heuristic_route(question: str) -> RoutingDecision:
    q = question.lower()
    sql_hit = any(h in q for h in _SQL_HINTS)
    rag_hit = any(h in q for h in _RAG_HINTS)

    if sql_hit and rag_hit:
        return RoutingDecision("hybrid", 0.5, "Question contains both data and policy keywords.")
    if sql_hit:
        return RoutingDecision("sql", 0.55, "Question matches patient-data keyword patterns.")
    if rag_hit:
        return RoutingDecision("rag", 0.55, "Question matches policy/procedure keyword patterns.")
    return RoutingDecision("clarify", 0.3, "No LLM configured and no keyword pattern matched confidently.")


def classify(question: str, history: list[ConversationTurn], llm: LLMClient) -> RoutingDecision:
    if not llm.available:
        return _heuristic_route(question)

    response = llm.complete(
        system=CLASSIFIER_SYSTEM_PROMPT,
        user=CLASSIFIER_USER_TEMPLATE.format(history=_format_history(history), question=question),
        temperature=0.0,
        max_tokens=200,
    )
    if response is None:
        return _heuristic_route(question)

    text = re.sub(r"^```(json)?|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
        route = parsed["route"]
        if route not in ("sql", "rag", "hybrid", "clarify", "unsupported"):
            raise ValueError(f"Unknown route '{route}'")
        return RoutingDecision(
            route=route,
            confidence=float(parsed.get("confidence", 0.7)),
            reasoning=parsed.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return _heuristic_route(question)
