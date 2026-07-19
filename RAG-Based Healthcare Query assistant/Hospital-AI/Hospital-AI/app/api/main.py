"""
FastAPI application entrypoint.

Startup sequence (see `lifespan`):
1. Ensure the SQLite database exists; run the CSV ETL if the admissions
   table is empty (so `docker run` / first `uvicorn` launch works with
   zero manual setup steps).
2. Build the RAG vector index from the policy documents.
3. Construct the Orchestrator (which owns the SQL and RAG agents) once,
   as a module-level singleton reused across requests.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api.routes import chat, health
from app.config import get_settings
from app.database.session import engine, init_db
from app.orchestrator.graph import Orchestrator
from app.rag.retriever import Retriever
from app.utils.logging_config import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

orchestrator: Orchestrator | None = None
retriever: Retriever | None = None


def _ensure_database_loaded() -> None:
    init_db()
    with engine.connect() as conn:
        try:
            count = conn.execute(text("SELECT COUNT(*) FROM admissions")).scalar_one()
        except Exception:
            count = 0
    if count == 0:
        logger.info("admissions table is empty; running ETL from raw CSV.")
        from app.database.load_data import load

        load()
    else:
        logger.info("admissions table already loaded (%d rows).", count)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, retriever

    logger.info("Starting up %s (environment=%s, llm_provider=%s)", settings.app_name, settings.environment, settings.llm_provider)

    _ensure_database_loaded()

    retriever = Retriever(settings)
    if Retriever and settings.vector_store_dir.exists():
        try:
            from app.rag.vector_store import VectorStore

            if VectorStore.exists(settings.vector_store_dir):
                logger.info("Loading persisted RAG index from %s", settings.vector_store_dir)
                from app.rag.embeddings import get_embedding_backend

                retriever.store = VectorStore.load(settings.vector_store_dir)
                retriever.backend = get_embedding_backend()
                # TF-IDF must be refit to encode new queries (see retriever.py docstring);
                # sentence-transformers needs no fitting. Cheap either way at this corpus size.
                from app.rag.loader import load_documents
                from app.rag.chunker import chunk_documents

                docs = load_documents(settings.documents_dir)
                chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
                retriever.backend.fit([c.text for c in chunks])
        except Exception:
            logger.warning("Failed to load persisted RAG index; rebuilding.", exc_info=True)

    if retriever.store is None:
        retriever.build(persist=True)

    orchestrator = Orchestrator(engine, retriever)
    logger.info("Startup complete.")

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    description="Multi-agent hospital assistant: NL2SQL over patient admissions + RAG over hospital policy documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. This has been logged."},
    )


app.include_router(health.router)
app.include_router(chat.router)


@app.get("/")
def root():
    return {"service": settings.app_name, "docs": "/docs", "health": "/health"}
