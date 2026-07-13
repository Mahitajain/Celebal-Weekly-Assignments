"""
cli.py
------
Simple command-line interface for the Document Question Answering (RAG) system.

Examples:
    # Build an index from one or more files, then ask questions interactively
    python cli.py --files sample_data/sample.txt

    # Use a whole folder of PDFs/notes
    python cli.py --dir sample_data/

    # Use Claude for generation (needs ANTHROPIC_API_KEY set in your env)
    python cli.py --files sample_data/sample.txt --llm anthropic

    # Ask a single question non-interactively
    python cli.py --files sample_data/sample.txt --question "What is RAG?"
"""

import argparse
import sys
from src.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Document Question Answering System (RAG)")
    parser.add_argument("--files", nargs="+", help="Path(s) to PDF/TXT/MD files")
    parser.add_argument("--dir", help="Directory of documents to load instead of --files")
    parser.add_argument("--embedding", default="sentence-transformers",
                         choices=["sentence-transformers", "tfidf"],
                         help="Embedding backend (default: sentence-transformers)")
    parser.add_argument("--llm", default="extractive",
                         choices=["anthropic", "openai", "extractive"],
                         help="Generation backend (default: extractive, no API key needed)")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--question", help="Ask a single question and exit (non-interactive)")
    args = parser.parse_args()

    if not args.files and not args.dir:
        print("Provide --files or --dir. See --help.")
        sys.exit(1)

    rag = RAGPipeline(embedding_backend=args.embedding, llm_backend=args.llm, top_k=args.top_k)

    if args.dir:
        rag.build_index_from_directory(args.dir)
    else:
        rag.build_index(args.files)

    def answer_and_print(question: str):
        answer, sources = rag.ask(question)
        print("\nAnswer:\n" + answer)
        print("\nSources:")
        for s in sources:
            print(f"  - {s['source']} (chunk {s['chunk_index']}, score={s['score']}): {s['excerpt']}...")
        print()

    if args.question:
        answer_and_print(args.question)
        return

    print("\nIndex built. Ask questions about your document(s) (type 'exit' to quit).\n")
    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("exit", "quit"):
            break
        answer_and_print(question)


if __name__ == "__main__":
    main()
