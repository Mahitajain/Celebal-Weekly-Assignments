"""
embed_store.py
--------------
Stages 3 & 4 of the RAG pipeline: Embedding Creation + Vector Database.

Two embedding backends are supported behind one interface:

  - "sentence-transformers" (default, recommended): dense semantic embeddings
    using a local MiniLM model. Best retrieval quality. Requires the model
    to be downloaded from Hugging Face the first time (needs internet).

  - "tfidf": a classic sparse bag-of-words embedder (scikit-learn). No
    internet or model download required, so it's used automatically as a
    fallback if sentence-transformers / the model download isn't available.
    Retrieval quality is lower (keyword-based, not semantic) but the whole
    pipeline still works end-to-end offline.

Vectors are indexed with FAISS for fast cosine-similarity search.
"""

import pickle
import numpy as np
import faiss

from .chunking import Chunk


class Embedder:
    """Wraps either a sentence-transformers model or a TF-IDF vectorizer
    behind a single .encode(list[str]) -> np.ndarray interface."""

    def __init__(self, backend: str = "sentence-transformers", model_name: str = "all-MiniLM-L6-v2"):
        self.backend = backend
        self.model_name = model_name
        self._model = None
        self._vectorizer = None  # only used for tfidf
        self.dim = None

        if backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name)
                self.dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"[embed_store] Falling back to TF-IDF embeddings "
                      f"(sentence-transformers unavailable: {e})")
                self.backend = "tfidf"

        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(max_features=4096)

    def fit(self, texts: list[str]):
        """Only needed for TF-IDF (must fit vocabulary on the corpus first)."""
        if self.backend == "tfidf":
            self._vectorizer.fit(texts)
            self.dim = len(self._vectorizer.vocabulary_)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend == "sentence-transformers":
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype="float32")
        else:
            vecs = self._vectorizer.transform(texts).toarray().astype("float32")
            # normalize for cosine similarity via inner product
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms


class VectorStore:
    """FAISS-backed vector database mapping chunk embeddings -> chunk metadata."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embedder.fit(texts)
        vectors = self.embedder.encode(texts)
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine (vectors normalized)
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 4) -> list[tuple[Chunk, float]]:
        q_vec = self.embedder.encode([query])
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path_prefix: str):
        faiss.write_index(self.index, f"{path_prefix}.faiss")
        with open(f"{path_prefix}.meta.pkl", "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "backend": self.embedder.backend,
                "model_name": self.embedder.model_name,
                "vectorizer": self.embedder._vectorizer,
            }, f)

    @classmethod
    def load(cls, path_prefix: str) -> "VectorStore":
        with open(f"{path_prefix}.meta.pkl", "rb") as f:
            meta = pickle.load(f)
        embedder = Embedder(backend=meta["backend"], model_name=meta["model_name"])
        embedder._vectorizer = meta["vectorizer"]
        store = cls(embedder)
        store.chunks = meta["chunks"]
        store.index = faiss.read_index(f"{path_prefix}.faiss")
        return store
