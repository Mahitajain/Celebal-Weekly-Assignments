"""
app.py
------
Streamlit UI for the Document Question Answering (RAG) system.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")
st.title("📄 Document Question Answering System (RAG)")
st.caption("Upload a document, then ask questions. Answers are grounded in retrieved excerpts, not the model's memory.")

with st.sidebar:
    st.header("Settings")
    embedding_backend = st.selectbox(
        "Embedding model", ["sentence-transformers", "tfidf"],
        help="sentence-transformers gives much better semantic retrieval, but needs internet "
             "access the first time to download the model. tfidf works fully offline.",
    )
    llm_backend = st.selectbox(
        "Answer generation", ["extractive", "anthropic", "openai"],
        help="anthropic/openai need the matching API key set as an environment variable "
             "(ANTHROPIC_API_KEY / OPENAI_API_KEY). extractive needs no API key.",
    )
    top_k = st.slider("Chunks to retrieve", 1, 8, 4)
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 800, step=100)

    if llm_backend != "extractive":
        env_var = "ANTHROPIC_API_KEY" if llm_backend == "anthropic" else "OPENAI_API_KEY"
        if not os.environ.get(env_var):
            st.warning(f"{env_var} is not set. This backend will silently fall back to extractive mode.")

uploaded_files = st.file_uploader(
    "Upload document(s)", type=["pdf", "txt", "md"], accept_multiple_files=True
)

use_sample = st.checkbox("Use bundled sample document instead", value=not uploaded_files)

if "rag" not in st.session_state:
    st.session_state.rag = None
    st.session_state.indexed_files = None

build_clicked = st.button("Build / rebuild index", type="primary")

if build_clicked:
    file_paths = []
    tmp_dir = tempfile.mkdtemp()

    if use_sample:
        file_paths = ["sample_data/sample.txt"]
    elif uploaded_files:
        for f in uploaded_files:
            path = os.path.join(tmp_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            file_paths.append(path)
    else:
        st.error("Upload a document or check 'Use bundled sample document'.")
        st.stop()

    with st.spinner("Ingesting documents, chunking, and building the vector index..."):
        rag = RAGPipeline(
            embedding_backend=embedding_backend,
            llm_backend=llm_backend,
            chunk_size=chunk_size,
            top_k=top_k,
        )
        rag.build_index(file_paths)
        st.session_state.rag = rag
        st.session_state.indexed_files = [os.path.basename(p) for p in file_paths]

    st.success(f"Index built from: {', '.join(st.session_state.indexed_files)} "
               f"({len(rag.store.chunks)} chunks)")

if st.session_state.rag:
    st.divider()
    question = st.text_input("Ask a question about your document(s)")

    if question:
        with st.spinner("Retrieving relevant context and generating an answer..."):
            answer, sources = st.session_state.rag.ask(question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved context (what the answer is grounded in)")
        for s in sources:
            with st.expander(f"{s['source']} — chunk {s['chunk_index']} (similarity {s['score']})"):
                st.write(s["excerpt"] + "...")
else:
    st.info("Upload a document (or use the sample) and click 'Build / rebuild index' to get started.")
