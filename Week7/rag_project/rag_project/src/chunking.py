"""
chunking.py
-----------
Stage 2 of the RAG pipeline: Text Chunking.

Splits raw document text into overlapping chunks. Chunking improves
retrieval accuracy: chunks that are too large dilute the embedding
with irrelevant content, chunks that are too small lose context.

Strategy used here: sentence-aware sliding window. We split on sentence
boundaries first, then greedily pack sentences into chunks of ~chunk_size
characters, with chunk_overlap characters repeated between consecutive
chunks so that context isn't lost at the boundary.
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    chunk_index: int


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list[str]:
    # Also split on double newlines (paragraph breaks) to avoid giant sentences
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sentences = []
    for para in paragraphs:
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip())
    return sentences


def chunk_text(source: str, text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[Chunk]:
    """
    Greedily pack sentences into chunks of approximately `chunk_size` characters,
    carrying `chunk_overlap` characters of trailing context into the next chunk.
    """
    sentences = split_into_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    idx = 0

    def flush():
        nonlocal current, current_len, idx
        if not current:
            return
        chunk_text_str = " ".join(current).strip()
        if chunk_text_str:
            chunks.append(Chunk(
                id=f"{source}::chunk{idx}",
                source=source,
                text=chunk_text_str,
                chunk_index=idx,
            ))
            idx += 1

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            flush()
            # carry overlap: keep trailing sentences that fit in chunk_overlap
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)
            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += len(sentence)

    flush()
    return chunks


def chunk_documents(documents, chunk_size: int = 800, chunk_overlap: int = 150) -> list[Chunk]:
    """Chunk a list of RawDocument objects (from ingest.py) into a flat list of Chunks."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc.source, doc.text, chunk_size, chunk_overlap))
    return all_chunks
