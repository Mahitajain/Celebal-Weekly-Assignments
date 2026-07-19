from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=128)


class Citation(BaseModel):
    document: str
    section_number: str | None = None
    section_title: str | None = None
    score: float


class SQLDetail(BaseModel):
    sql: str | None = None
    explanation: str | None = None
    columns: list[str] = []
    rows: list[dict] = []
    row_count: int = 0
    used_llm: bool = False


class RAGDetail(BaseModel):
    citations: list[Citation] = []
    used_llm: bool = False


class ChatResponse(BaseModel):
    answer: str
    route: str
    route_confidence: float
    route_reasoning: str
    sql_detail: SQLDetail | None = None
    rag_detail: RAGDetail | None = None
    session_id: str


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_available: bool
    embedding_backend: str
    admissions_row_count: int
    documents_indexed: int
