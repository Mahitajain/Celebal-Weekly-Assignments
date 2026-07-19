"""
FAISS-backed vector store.

We use `IndexFlatIP` (exact inner-product search) over L2-normalized
vectors, which is mathematically equivalent to cosine similarity. For a
few thousand policy-document chunks, exact search is fast enough that an
approximate index (IVF/HNSW) would only add complexity without a real
latency win -- that trade-off is documented in docs/RAG_PIPELINE.md and
is the kind of "choose what real systems use, and explain why" judgment
call the project calls for. Swapping in `IndexHNSWFlat` later for a much
larger corpus is a one-line change since the rest of the interface
(`add`, `search`) doesn't change.

ChromaDB is a reasonable alternative for this same job (it bundles
storage + metadata filtering out of the box); FAISS was chosen here to
keep the dependency footprint small and the on-disk format transparent
(a flat index file + a JSON sidecar for metadata), which matters for a
project meant to run with `pip install -r requirements.txt` and nothing
else.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from app.rag.chunker import Chunk


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    doc_title: str
    section_number: str | None
    section_title: str | None
    text: str
    score: float


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        assert vectors.shape[0] == len(chunks)
        assert vectors.shape[1] == self.dim
        self.index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        if self.index.ntotal == 0:
            return []
        query_vector = query_vector.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    section_number=chunk.section_number,
                    section_title=chunk.section_title,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return results

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        metadata = [asdict(c) for c in self._chunks]
        (directory / "chunks.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (directory / "meta.json").write_text(json.dumps({"dim": self.dim}), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        store = cls(dim=meta["dim"])
        store.index = faiss.read_index(str(directory / "index.faiss"))
        raw_chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        store._chunks = [Chunk(**c) for c in raw_chunks]
        return store

    @staticmethod
    def exists(directory: Path) -> bool:
        directory = Path(directory)
        return (directory / "index.faiss").exists() and (directory / "chunks.json").exists()
