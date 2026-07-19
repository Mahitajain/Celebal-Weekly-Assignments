from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import ChatRequest, ChatResponse, RAGDetail, SQLDetail
from app.orchestrator.memory import get_memory
from app.utils.cache import get_query_cache
from app.utils.logging_config import get_logger

router = APIRouter(prefix="/api/v1", tags=["chat"])
logger = get_logger(__name__)


def get_orchestrator():
    # Imported lazily / injected from app.api.main to avoid a circular import
    # and to make this endpoint trivially testable with a stub orchestrator.
    from app.api.main import orchestrator

    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator is not initialized yet.")
    return orchestrator


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, orch=Depends(get_orchestrator)) -> ChatResponse:
    memory = get_memory()
    cache = get_query_cache()

    cache_key = f"{request.session_id}:{request.message.strip().lower()}"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info("Cache hit for session=%s", request.session_id)
        return cached

    history = memory.get(request.session_id)

    try:
        state = orch.run(request.message, history=history)
    except Exception:
        logger.exception("Orchestrator run failed")
        raise HTTPException(status_code=500, detail="The assistant hit an internal error processing that question.")

    memory.append(request.session_id, "user", request.message)
    memory.append(request.session_id, "assistant", state.get("final_answer", ""))

    sql_detail = None
    if state.get("sql_result"):
        sr = state["sql_result"]
        sql_detail = SQLDetail(
            sql=sr.get("sql"),
            explanation=sr.get("explanation"),
            columns=sr.get("columns", []),
            rows=sr.get("rows", []),
            row_count=sr.get("row_count", 0),
            used_llm=sr.get("used_llm", False),
        )

    rag_detail = None
    if state.get("rag_result"):
        rr = state["rag_result"]
        rag_detail = RAGDetail(citations=rr.get("citations", []), used_llm=rr.get("used_llm", False))

    response = ChatResponse(
        answer=state.get("final_answer", ""),
        route=state.get("route", "unsupported"),
        route_confidence=state.get("route_confidence", 0.0),
        route_reasoning=state.get("route_reasoning", ""),
        sql_detail=sql_detail,
        rag_detail=rag_detail,
        session_id=request.session_id,
    )
    cache.set(cache_key, response)
    return response


@router.delete("/chat/{session_id}")
def clear_session(session_id: str) -> dict:
    get_memory().clear(session_id)
    return {"cleared": session_id}
