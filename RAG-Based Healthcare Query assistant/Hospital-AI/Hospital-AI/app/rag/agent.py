"""
RAG Agent.

Pipeline: embed query -> retrieve top-k chunks -> (if best score is below
`similarity_threshold`, decline rather than guess) -> generate a grounded
answer citing the specific document sections retrieved -> if no LLM is
configured, fall back to returning the retrieved section(s) verbatim as an
extractive answer.

Hallucination prevention has three layers here:
1. Similarity threshold gate: if nothing retrieved is actually relevant,
   the agent says so instead of asking the LLM to improvise.
2. The generation prompt instructs the model to answer *only* from the
   provided context and to say so explicitly if the context doesn't cover
   the question.
3. Every answer carries the source chunks it was generated from, so a
   human can verify the claim against the cited section directly -- this
   is the same "show your work" pattern used in the copyright-safe
   citation approach elsewhere in this codebase.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import get_settings
from app.llm.client import LLMClient, get_llm_client
from app.rag.retriever import Retriever
from app.rag.vector_store import SearchResult
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

RAG_SYSTEM_PROMPT = """You are a hospital policy assistant. You answer hospital staff questions \
using ONLY the policy document excerpts provided below. 

RULES:
- Answer only from the provided context. Do not use outside knowledge about hospital policy.
- If the context does not contain the answer, say clearly that the policy documents provided don't \
cover that question, rather than guessing.
- Be concise and direct -- 2-5 sentences unless the question needs a list.
- When you state a specific rule (a number, a time window, a requirement), mention which document/section \
it came from inline, e.g. "(Visitor Policy, Section 3)".

CONTEXT:
{context}
"""

RAG_USER_PROMPT = """Question: {question}"""


@dataclass
class Citation:
    doc_title: str
    section_number: str | None
    section_title: str | None
    score: float


@dataclass
class RAGAgentResult:
    success: bool
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    used_llm: bool = False
    elapsed_ms: float = 0.0
    error: str | None = None


class RAGAgent:
    def __init__(self, retriever: Retriever, llm: LLMClient | None = None):
        self.retriever = retriever
        self.llm = llm or get_llm_client()
        self.settings = get_settings()

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        blocks = []
        for r in results:
            label = r.doc_title
            if r.section_number:
                label += f" \u2014 Section {r.section_number}: {r.section_title}"
            blocks.append(f"[{label}]\n{r.text}")
        return "\n\n---\n\n".join(blocks)

    def _extractive_answer(self, results: list[SearchResult]) -> str:
        top = results[0]
        label = top.doc_title
        if top.section_number:
            label += f", Section {top.section_number} ({top.section_title})"
        # Strip the header line we prepended during chunking before showing it as the "answer".
        body = top.text.split("]\n", 1)[-1]
        return f"From {label}:\n\n{body}"

    def answer(self, question: str) -> RAGAgentResult:
        t0 = time.time()
        results = self.retriever.retrieve(question)

        if not results or results[0].score < self.settings.similarity_threshold:
            return RAGAgentResult(
                success=False,
                error=(
                    "None of the hospital policy documents on file appear to address this question. "
                    "Try rephrasing, or this may need to be routed to a person."
                ),
                elapsed_ms=(time.time() - t0) * 1000,
            )

        citations = [
            Citation(doc_title=r.doc_title, section_number=r.section_number, section_title=r.section_title, score=r.score)
            for r in results
        ]

        if self.llm.available:
            context = self._format_context(results)
            system = RAG_SYSTEM_PROMPT.format(context=context)
            response = self.llm.complete(system=system, user=RAG_USER_PROMPT.format(question=question), temperature=0.1)
            if response is not None:
                return RAGAgentResult(
                    success=True,
                    answer=response.text.strip(),
                    citations=citations,
                    used_llm=True,
                    elapsed_ms=round((time.time() - t0) * 1000, 1),
                )
            logger.warning("LLM call failed for RAG agent; falling back to extractive answer.")

        return RAGAgentResult(
            success=True,
            answer=self._extractive_answer(results),
            citations=citations,
            used_llm=False,
            elapsed_ms=round((time.time() - t0) * 1000, 1),
        )
