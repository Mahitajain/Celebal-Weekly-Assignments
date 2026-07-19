"""
Pluggable embedding backends.

- ``sentence-transformers``: dense semantic embeddings (default model
  `all-MiniLM-L6-v2`, 384-dim). This is the recommended backend for real
  deployments -- it captures paraphrase/synonym similarity ("visitor
  hours" ~ "when can family visit") that lexical matching misses.
- ``tfidf``: a scikit-learn TF-IDF vectorizer. No model download, no GPU,
  runs anywhere pip works. It is a legitimate production choice too for
  small, vocabulary-stable corpora like a fixed set of policy documents
  (which is exactly this project's document set) -- the trade-off is it
  won't match true synonyms it never saw in the corpus.

Both backends implement the same `EmbeddingBackend` interface so the rest
of the RAG pipeline (vector store, retriever) never needs to know which
one is active.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.config import get_settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingBackend(ABC):
    name: str
    dim: int

    @abstractmethod
    def fit(self, corpus: list[str]) -> None:
        """Fit any corpus-dependent state (e.g. TF-IDF vocabulary). No-op for pretrained models."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings."""


class SentenceTransformerBackend(EmbeddingBackend):
    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # local import: heavy optional dep

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def fit(self, corpus: list[str]) -> None:
        return  # pretrained model, nothing to fit

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")


class TfidfBackend(EmbeddingBackend):
    """Dependency-light fallback: TF-IDF vectors, L2-normalized for cosine similarity via inner product."""

    name = "tfidf"

    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self._fitted = False
        self.dim = max_features

    def fit(self, corpus: list[str]) -> None:
        self._vectorizer.fit(corpus)
        self.dim = len(self._vectorizer.vocabulary_)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfBackend.fit() must be called on the corpus before encode().")
        matrix = self._vectorizer.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def get_embedding_backend() -> EmbeddingBackend:
    settings = get_settings()
    if settings.embedding_backend == "sentence-transformers":
        try:
            return SentenceTransformerBackend(settings.embedding_model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; falling back to TF-IDF embeddings. "
                "Install `sentence-transformers` and set EMBEDDING_BACKEND=sentence-transformers "
                "for production-grade semantic search."
            )
    return TfidfBackend()
