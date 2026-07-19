"""Load hospital policy documents from disk with source metadata attached.

Each markdown file is expected to start with an H1 title and contain
`## N. Section Title` headers -- this loader captures that structure so
downstream chunking can carry section-level citations rather than just
"chunk 7 of some file."
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    title: str
    source_path: str
    text: str


_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def load_documents(documents_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(Path(documents_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_match = _H1.search(text)
        title = title_match.group(1).strip() if title_match else path.stem.replace("_", " ").title()
        documents.append(
            Document(doc_id=path.stem, title=title, source_path=str(path), text=text)
        )
    return documents
