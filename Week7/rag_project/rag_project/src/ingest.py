"""
ingest.py
---------
Stage 1 of the RAG pipeline: Document Ingestion.

Loads documents (PDF, TXT, MD) from disk and converts them into raw text,
tagged with their source filename (used later for citations).
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class RawDocument:
    source: str   # filename
    text: str     # full extracted text


def load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        content = page.extract_text() or ""
        pages.append(content)
    return "\n".join(pages)


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document(path: str) -> RawDocument:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {path}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        text = load_pdf(p)
    elif suffix in (".txt", ".md"):
        text = load_txt(p)
    else:
        raise ValueError(f"Unsupported file type: {suffix} (supported: .pdf, .txt, .md)")

    # Basic cleanup: collapse excessive whitespace
    text = "\n".join(line.strip() for line in text.splitlines())
    text = "\n".join(filter(None, text.split("\n")))

    return RawDocument(source=p.name, text=text)


def load_documents(paths: list[str]) -> list[RawDocument]:
    """Load multiple documents (e.g. a whole folder of PDFs/notes)."""
    docs = []
    for path in paths:
        try:
            docs.append(load_document(path))
        except Exception as e:
            print(f"[ingest] Skipping {path}: {e}")
    return docs


def load_directory(dir_path: str) -> list[RawDocument]:
    """Load every supported file in a directory."""
    p = Path(dir_path)
    paths = [str(f) for f in p.iterdir() if f.suffix.lower() in (".pdf", ".txt", ".md")]
    return load_documents(paths)
