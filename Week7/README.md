# Document Question Answering System (RAG)

A Retrieval-Augmented Generation pipeline that answers questions from your own
documents (PDFs, notes, resumes, research papers, books, etc.) instead of relying
only on a language model's internal, possibly-outdated memory.

## How it works

```
 PDF/TXT files
      │  ingest.py
      ▼
 Raw text
      │  chunking.py  (sentence-aware sliding window, with overlap)
      ▼
 Text chunks
      │  embed_store.py (sentence-transformers or TF-IDF)
      ▼
 Embeddings ──► FAISS vector index
      │
      │  user question ──► embedded ──► similarity search
      ▼
 Top-k relevant chunks
      │  generator.py
      ▼
 Answer grounded in retrieved context (Claude / GPT / offline extractive)
```

| Stage | File | What it does |
|---|---|---|
| 1. Document Ingestion | `src/ingest.py` | Loads PDF/TXT/MD files into raw text |
| 2. Text Chunking | `src/chunking.py` | Splits text into overlapping ~800-char chunks |
| 3. Embedding Creation | `src/embed_store.py` | Converts chunks into vectors |
| 4. Vector Database | `src/embed_store.py` | FAISS index for similarity search |
| 5-6. Query + Retrieval | `src/rag_pipeline.py` | Embeds the question, retrieves top-k chunks |
| 7. Answer Generation | `src/generator.py` | LLM generates an answer from retrieved context |

## Setup

```bash
pip install -r requirements.txt
```

If you only want to test the pipeline without any API keys or model downloads,
you can skip `sentence-transformers`/`anthropic`/`openai` — the system
automatically falls back to a TF-IDF embedder and an offline extractive
answerer (see "Backends" below).

## Quick start (CLI)

```bash
# Uses the bundled sample document about RAG concepts
python cli.py --files sample_data/sample.txt --question "How does RAG work?"

# Interactive mode - ask multiple questions
python cli.py --files sample_data/sample.txt

# Your own PDF/notes
python cli.py --files my_resume.pdf

# A whole folder of documents
python cli.py --dir my_notes/

# Best quality: semantic embeddings + Claude for generation
export ANTHROPIC_API_KEY=sk-ant-...
python cli.py --files my_document.pdf --embedding sentence-transformers --llm anthropic
```

## Quick start (Web UI)

```bash
streamlit run app.py
```

Upload a document in the browser, click "Build / rebuild index", and start
asking questions. The UI shows both the generated answer and the exact
retrieved excerpts it's grounded in.

## Backends

**Embeddings**
- `sentence-transformers` (default) — dense semantic embeddings using
  `all-MiniLM-L6-v2`. Best retrieval quality. Downloads the model from
  Hugging Face on first use (needs internet once, then it's cached).
- `tfidf` — classic keyword-based sparse vectors (scikit-learn). No
  internet or model download needed; use this if you're offline or want a
  lightweight demo.

**Generation**
- `anthropic` — Claude, via `ANTHROPIC_API_KEY`. Recommended for best answers.
- `openai` — GPT, via `OPENAI_API_KEY`.
- `extractive` (default) — no API key needed. Picks and stitches together the
  most relevant sentences from the retrieved chunks. Good for testing the
  retrieval pipeline itself; lower answer quality than a real LLM.

Both dimensions are independent, so e.g. `--embedding tfidf --llm anthropic`
(offline retrieval + real LLM answers) works fine too.

## Project structure

```
rag_project/
├── src/
│   ├── ingest.py         # Stage 1: document loading
│   ├── chunking.py       # Stage 2: text chunking
│   ├── embed_store.py    # Stages 3-4: embeddings + FAISS vector store
│   ├── generator.py      # Stage 7: LLM / extractive answer generation
│   └── rag_pipeline.py   # Wires everything together
├── cli.py                # Command-line interface
├── app.py                # Streamlit web UI
├── sample_data/
│   └── sample.txt        # Demo document so you can try it immediately
├── requirements.txt
└── .env.example
```

## Example

```
$ python cli.py --files sample_data/sample.txt --question "What are the limitations of RAG?"

Answer:
RAG systems are only as good as their retrieval step. If the retriever fails to
find the relevant chunk, the language model has no way to answer correctly, even
if the information exists somewhere in the document collection. This is why
techniques like hybrid search (combining keyword search with vector search) and
re-ranking are often added to improve retrieval quality.

Sources:
  - sample.txt (chunk 4, score=0.71): ...
```

## Extending this project

Ideas from the assignment brief, and how to try them here:
- **Better chunking**: tune `chunk_size`/`chunk_overlap` in `RAGPipeline(...)`, or
  swap the sentence-based splitter in `chunking.py` for a semantic/recursive one.
- **Different embedding models**: pass any Hugging Face sentence-transformers
  model name into `Embedder(model_name=...)`, e.g. `all-mpnet-base-v2`.
- **Hybrid search**: combine the `tfidf` and `sentence-transformers` scores in
  `VectorStore.search()` for keyword + semantic retrieval.
- **Re-ranking**: add a cross-encoder re-ranking pass over the top-k results
  in `rag_pipeline.py` before generation.
- **Different LLMs**: `generator.py` is a thin wrapper — add a new `_generate_x`
  method for any other API or a local Hugging Face model.
