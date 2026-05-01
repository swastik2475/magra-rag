

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def setup(force_rebuild: bool = False) -> None:
    """Ingest corpus and build FAISS index."""
    from ingest import ingest_corpus
    from retriever import build_index

    print("[setup] Ingesting corpus …")
    chunks = ingest_corpus(
        contract_pdf = os.getenv("CONTRACT_PDF",  "./data/contract.pdf"),
        schedule_csv = os.getenv("SCHEDULE_CSV",  "./data/schedule-construction.csv"),
        sov_pdf      = os.getenv("SOV_PDF",       "./data/schedule-of-values.pdf"),
    )

    print("[setup] Building FAISS vector index …")
    build_index(
        chunks,
        persist_dir   = os.getenv("FAISS_DIR", "./faiss_db"),
        force_rebuild = force_rebuild,
    )
    print("[setup] Done! Run: python main.py --ask 'your question'")


def ask(question: str, verbose: bool = False, top_k: int = 8) -> None:
    """Retrieve context and answer a question."""
    from retriever import get_index, retrieve, load_chunk_lookup
    from qa import answer, print_answer

    # Load FAISS index
    try:
        index = get_index(persist_dir=os.getenv("FAISS_DIR", "./faiss_db"))
    except FileNotFoundError as e:
        print(f"[error] {e}")
        sys.exit(1)

    # Load chunk lookup for keyword boost
    cache_path   = "./data/corpus_chunks.json"
    chunk_lookup = load_chunk_lookup(cache_path) if Path(cache_path).exists() else None

    # Retrieve
    chunks = retrieve(question, index, top_k=top_k, chunk_lookup=chunk_lookup)

    if verbose:
        print(f"\n[Retrieved {len(chunks)} chunks]")
        for i, c in enumerate(chunks, 1):
            print(f"  {i}. {c['citation']} (dist={c['distance']})")
            print(f"     {c['text'][:120].strip()} …")

    # Answer
    result = answer(question, chunks)
    print_answer(question, result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Magra RAG – Construction Contract Q&A (FAISS)"
    )
    parser.add_argument("--setup",         action="store_true", help="Ingest corpus + build index")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild index even if it exists")
    parser.add_argument("--ask",           type=str,            help="Ask a question")
    parser.add_argument("--verbose",       action="store_true", help="Show retrieved chunks")
    parser.add_argument("--top-k",         type=int, default=8, help="Number of chunks to retrieve")
    args = parser.parse_args()

    if args.setup:
        setup(force_rebuild=args.force_rebuild)
    elif args.ask:
        ask(args.ask, verbose=args.verbose, top_k=args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
