"""
rag_pipeline.py
----------------
Ties every stage together into one RAGPipeline class:

  ingest -> chunk -> embed -> store -> retrieve -> augment -> generate

Usage:
    from src.rag_pipeline import RAGPipeline

    rag = RAGPipeline(embedding_backend="sentence-transformers", llm_backend="anthropic")
    rag.build_index(["mydoc.pdf"])
    answer, sources = rag.ask("What is the main idea of the document?")
"""

from .ingest import load_documents, load_directory
from .chunking import chunk_documents
from .embed_store import Embedder, VectorStore
from .generator import Generator


class RAGPipeline:
    def __init__(
        self,
        embedding_backend: str = "sentence-transformers",
        llm_backend: str = "extractive",
        llm_model: str | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 4,
    ):
        self.embedder = Embedder(backend=embedding_backend)
        self.generator = Generator(backend=llm_backend, model=llm_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.store: VectorStore | None = None

    # ---- Index building -------------------------------------------------

    def build_index(self, file_paths: list[str]):
        docs = load_documents(file_paths)
        self._build_from_docs(docs)

    def build_index_from_directory(self, dir_path: str):
        docs = load_directory(dir_path)
        self._build_from_docs(docs)

    def _build_from_docs(self, docs):
        if not docs:
            raise ValueError("No documents were loaded. Check your file paths.")
        chunks = chunk_documents(docs, self.chunk_size, self.chunk_overlap)
        print(f"[rag_pipeline] Loaded {len(docs)} document(s) -> {len(chunks)} chunks.")
        self.store = VectorStore(self.embedder)
        self.store.build(chunks)

    def save_index(self, path_prefix: str):
        if not self.store:
            raise RuntimeError("No index built yet.")
        self.store.save(path_prefix)

    def load_index(self, path_prefix: str):
        self.store = VectorStore.load(path_prefix)
        self.embedder = self.store.embedder

    # ---- Question answering ---------------------------------------------

    def retrieve(self, question: str):
        if not self.store:
            raise RuntimeError("No index built yet. Call build_index() first.")
        return self.store.search(question, top_k=self.top_k)

    def ask(self, question: str):
        """Returns (answer: str, sources: list[dict]) for a given question."""
        results = self.retrieve(question)
        context_chunks = [chunk.text for chunk, _ in results]
        answer = self.generator.generate(question, context_chunks)
        sources = [
            {"source": chunk.source, "chunk_index": chunk.chunk_index,
             "score": round(score, 4), "excerpt": chunk.text[:200]}
            for chunk, score in results
        ]
        return answer, sources
