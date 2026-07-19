from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app.api.schemas import HealthResponse
from app.config import get_settings
from app.database.session import engine
from app.llm.client import get_llm_client

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    llm = get_llm_client()

    try:
        with engine.connect() as conn:
            row_count = conn.execute(text("SELECT COUNT(*) FROM admissions")).scalar_one()
    except Exception:
        row_count = -1

    from app.api.main import retriever

    documents_indexed = 0
    if retriever is not None and retriever.store is not None:
        documents_indexed = len({c.doc_id for c in retriever.store._chunks})

    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        llm_available=llm.available,
        embedding_backend=settings.embedding_backend,
        admissions_row_count=row_count,
        documents_indexed=documents_indexed,
    )
