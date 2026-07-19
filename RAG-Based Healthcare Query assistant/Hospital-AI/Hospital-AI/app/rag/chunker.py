"""
Chunking strategy.

We chunk *within* section boundaries (`## N. Title`) rather than blindly
by character count across the whole document. Rationale (see
docs/RAG_PIPELINE.md for the full writeup):

- Hospital policy documents are already organized into short, self-
  contained numbered sections ("3. Number of Visitors"). Splitting mid-
  section would separate a rule from the context that makes it correct
  (e.g. "two visitors" without "on general wards" nearby).
- Section-aware chunks let us cite "Visitor Policy, Section 3" instead of
  an opaque "chunk 12", which is what the RAG requirement of "cite the
  document sections used" actually asks for.
- Chunk size (default 800 chars, ~150-200 tokens) was chosen empirically:
  most sections in these documents run 400-900 characters, so this keeps
  ~90% of sections whole while still capping the few long sections
  (Infection Control section 4, Pre-Surgery section 3) at a size that
  keeps retrieval precise and generation cheap. Long sections are split
  with a character overlap so a split point never destroys context that
  spans the boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.loader import Document

_SECTION_HEADER = re.compile(r"^##\s+(\d+)\.\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section_number: str | None
    section_title: str | None
    text: str


def _split_into_sections(document: Document) -> list[tuple[str | None, str | None, str]]:
    matches = list(_SECTION_HEADER.finditer(document.text))
    if not matches:
        return [(None, None, document.text)]

    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(document.text)
        section_number, section_title = m.group(1), m.group(2).strip()
        body = document.text[start:end].strip()
        sections.append((section_number, section_title, body))
    return sections


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return pieces


def chunk_documents(documents: list[Document], chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        for section_number, section_title, body in _split_into_sections(document):
            pieces = _split_long_text(body, chunk_size, overlap)
            for i, piece in enumerate(pieces):
                suffix = f"-{section_number or 'full'}" + (f".{i}" if len(pieces) > 1 else "")
                chunk_id = f"{document.doc_id}{suffix}"
                # Prepend lightweight header context so the embedding and the
                # LLM both see which document/section this text belongs to,
                # even after the surrounding markdown headers are stripped.
                header = f"[{document.title}. {document.title}"
                if section_title:
                    header += f" \u2014 Section {section_number}: {section_title}"
                header += "]\n"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        doc_title=document.title,
                        section_number=section_number,
                        section_title=section_title,
                        text=header + piece.strip(),
                    )
                )
    return chunks
