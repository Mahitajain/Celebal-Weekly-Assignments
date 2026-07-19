from __future__ import annotations

from app.rag.agent import RAGAgent


def test_retrieval_finds_relevant_document(retriever):
    results = retriever.retrieve("What is the hospital visitor policy?")
    assert results
    assert results[0].doc_id == "visitor_policy"


def test_retrieval_returns_scores_descending(retriever):
    results = retriever.retrieve("infection control isolation room precautions")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rag_agent_declines_irrelevant_question(retriever):
    agent = RAGAgent(retriever)
    result = agent.answer("What is the capital of France?")
    assert not result.success
    assert result.error


def test_rag_agent_answers_with_citation(retriever):
    agent = RAGAgent(retriever)
    result = agent.answer("What is required before surgery?")
    assert result.success
    assert result.citations
    assert result.citations[0].doc_title == "Pre-Surgery Requirements and Documentation"


def test_chunking_preserves_section_metadata(retriever):
    doc_ids = {c.doc_id for c in retriever.store._chunks}
    assert "emergency_discharge_procedure" in doc_ids
    section_numbers = {c.section_number for c in retriever.store._chunks if c.doc_id == "visitor_policy"}
    assert "1" in section_numbers and "9" in section_numbers
