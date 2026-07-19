# RAG Pipeline

## Documents

Six synthetic hospital policy documents in `data/documents/`: Visitor
Policy, Emergency Discharge Procedure, Infection Control Protocol,
Pre-Surgery Requirements, Patient Data Privacy Policy, Billing and
Insurance Policy. Each is markdown with a document ID, effective date,
owning department, and numbered `## N. Section Title` sections -- this
structure is not incidental, it's what makes section-level citation
possible (see Chunking below).

## Loading (`app/rag/loader.py`)

Reads every `.md` file in the documents directory, extracts the H1 title,
and returns a lightweight `Document` (id, title, source path, raw text).

## Chunking (`app/rag/chunker.py`)

**Strategy: chunk within section boundaries, not by blind character
count.** A regex matches `## N. Title` headers and chunks are cut at
those boundaries first; only sections that exceed `chunk_size` get
further split (with overlap, so a split point never fully severs
context). This is a deliberate choice over naive fixed-window chunking:

- These documents are already organized into short, self-contained
  numbered rules ("Section 3: Number of Visitors" is a complete,
  independently-meaningful unit). Splitting mid-section risks separating
  a rule from the qualifier that makes it correct (e.g. "two visitors"
  without "on general wards" nearby).
- The brief asks the RAG agent to "cite the document sections used."
  Section-aware chunking makes that a direct, honest citation ("Visitor
  Policy, Section 3") instead of an opaque "chunk #12 of file X."

**Chunk size (800 chars, ~150-200 tokens) and overlap (120 chars) were
chosen empirically** against this document set: most sections run
400-900 characters, so 800 keeps the large majority of sections whole,
while the overlap protects the handful of longer sections (Infection
Control Section 4, Pre-Surgery Section 3) from losing context at their
internal split point.

Every chunk is prefixed with a small header
(`[Visitor Policy. Visitor Policy — Section 3: Number of Visitors]`)
before embedding. The title is repeated once deliberately: for the
TF-IDF backend, this doubles the term weight of the document title in
that chunk's vector, which measurably improved retrieval precision for
whole-document queries like "what is the visitor policy" during
development (see `app/rag/chunker.py` inline comment and the retrieval
test in `tests/test_rag_pipeline.py`).

## Embeddings (`app/rag/embeddings.py`) -- pluggable backends

- **`sentence-transformers`** (`all-MiniLM-L6-v2`, 384-dim): the
  recommended backend for real deployments. Captures paraphrase/synonym
  similarity ("when can family visit" ~ "visitor hours") that pure
  lexical matching misses. Implemented and wired end-to-end but requires
  `pip install sentence-transformers` (pulls in `torch`), which was too
  heavy for this build environment's disk budget to install and verify
  -- the code path is real, not a stub, but ships untested in *this*
  environment. Enable with `EMBEDDING_BACKEND=sentence-transformers`.
- **`tfidf`** (scikit-learn `TfidfVectorizer`, unigrams+bigrams, English
  stopwords removed, sublinear TF scaling): the default, tested backend.
  No model download, no GPU, runs anywhere `pip` works. This is a
  legitimate production choice, not just a fallback, for a corpus like
  this one: a small, fixed, vocabulary-stable set of policy documents
  where exact terminology ("ASA physical status classification", "NPO")
  matters and won't drift. The honest trade-off: it will not match a
  true synonym it never saw in the corpus.

Both backends implement the same `EmbeddingBackend.fit()` /
`.encode()` interface so the rest of the pipeline never branches on
which one is active.

## Vector store (`app/rag/vector_store.py`)

FAISS `IndexFlatIP` (exact inner-product search) over L2-normalized
vectors, which is mathematically equivalent to cosine similarity. For a
few thousand chunks, exact search costs microseconds -- an approximate
index (IVF/HNSW) would add real complexity for no measurable latency
win at this scale. That trade-off is explicit in code comments so it's
clear it's a scale-aware decision, not an oversight; swapping in
`IndexHNSWFlat` for a much larger corpus later is a one-line change since
`add()`/`search()` don't change.

ChromaDB is a reasonable alternative that bundles storage + metadata
filtering; FAISS was chosen to keep the dependency footprint small and
the on-disk artifact transparent (a flat index file + a JSON sidecar),
which matters for `pip install -r requirements.txt` being the entire
setup story.

## Retrieval (`app/rag/retriever.py`)

`Retriever.build()` loads, chunks, fits the embedding backend on the
corpus, encodes, and indexes -- run once at API startup (or on demand via
`python -m app.rag.retriever` style scripting). The embedding backend is
**refit on every process start** rather than persisting fitted TF-IDF
vectorizer state: for six documents this costs well under a second, and
it sidesteps a real correctness trap -- a persisted vectorizer silently
goes stale the moment a document is edited, while "rebuild on startup"
is always correct by construction. The FAISS index itself is still
persisted to `vector_store/` so a restart doesn't require re-encoding
if the corpus hasn't changed (the file is regenerated if the corpus
*has* changed, since `build()` is what writes it).

## Hallucination prevention (three layers)

1. **Similarity gate**: if the top retrieval score is below
   `SIMILARITY_THRESHOLD` (default 0.15), the agent declines outright
   rather than asking the LLM to answer from irrelevant context.
2. **Constrained generation prompt**: the RAG system prompt explicitly
   instructs the model to answer only from the provided excerpts and to
   say so plainly if they don't cover the question, rather than filling
   gaps from general knowledge.
3. **Mandatory citations**: every successful answer returns the specific
   chunks (document + section + relevance score) it was generated from,
   surfaced in the API response and rendered as an expandable "Sources"
   panel in the UI -- so a human can verify any specific claim against
   the cited section directly, the same "show your work" principle used
   for citation handling elsewhere in this codebase.

## No-LLM fallback

If no LLM is configured, the RAG agent returns the single most relevant
retrieved section verbatim ("From Visitor Policy, Section 3: ...")
instead of a generated summary. This is intentionally extractive rather
than generative -- it can't hallucinate because it isn't generating
anything, only quoting the source it already decided was relevant.
