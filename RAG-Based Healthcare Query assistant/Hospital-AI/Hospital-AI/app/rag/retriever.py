"""
Retrieval pipeline glue: load documents -> chunk -> embed -> index -> search.

The embedding backend is refit on the document corpus every time the
process starts (cheap: a handful of policy documents, well under a
second even for TF-IDF) rather than persisting fitted vectorizer state.
This keeps the on-disk artifact set simple (just the FAISS index +
chunk metadata) and sidesteps a subtle correctness trap: a persisted
TF-IDF vectorizer silently goes stale the moment a document is added or
edited, while "rebuild on startup" is always correct by construction.
For the sentence-transformers backend this distinction doesn't matter at
all since there's no corpus-dependent fitting step.
"""
from __future__ import annotations

from pathlib import Path

from app.config import Settings, get_settings
from app.rag.chunker import Chunk, chunk_documents
from app.rag.embeddings import EmbeddingBackend, get_embedding_backend
from app.rag.loader import load_documents
from app.rag.vector_store import SearchResult, VectorStore
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.backend: EmbeddingBackend | None = None
        self.store: VectorStore | None = None
        self._chunks: list[Chunk] = []

    def build(self, persist: bool = True) -> dict:
        documents = load_documents(self.settings.documents_dir)
        if not documents:
            raise FileNotFoundError(f"No .md documents found in {self.settings.documents_dir}")

        chunks = chunk_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        self._chunks = chunks

        self.backend = get_embedding_backend()
        corpus = [c.text for c in chunks]
        self.backend.fit(corpus)
        vectors = self.backend.encode(corpus)

        self.store = VectorStore(dim=self.backend.dim)
        self.store.add(chunks, vectors)

        if persist:
            self.store.save(self.settings.vector_store_dir)

        stats = {
            "documents": len(documents),
            "chunks": len(chunks),
            "embedding_backend": self.backend.name,
            "dim": self.backend.dim,
        }
        logger.info("RAG index built: %s", stats)
        return stats

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        if self.backend is None or self.store is None:
            self.build()
        top_k = top_k or self.settings.retrieval_top_k
        query_vector = self.backend.encode([query])[0]
        return self.store.search(query_vector, top_k)


_retriever_singleton: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = Retriever()
        _retriever_singleton.build()
    return _retriever_singleton
