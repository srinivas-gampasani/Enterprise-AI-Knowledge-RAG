#!/usr/bin/env python3
"""
Ascension Via Christi RAG System — CLI Query Tool

Usage:
    cd backend && python ../scripts/query_cli.py
    cd backend && python ../scripts/query_cli.py --question "What is the sepsis protocol?"
"""
import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))

from app.services.rag_service import RAGService
from app.core.config import settings


def print_banner():
    print("\n" + "=" * 65)
    print("  ASCENSION VIA CHRISTI — AI Knowledge Assistant (CLI)")
    print("=" * 65)
    print(f"  Model: {settings.OPENAI_MODEL} | Embeddings: {settings.EMBEDDING_MODEL}")
    print(f"  Documents: {settings.DOCS_PATH}")
    print("=" * 65 + "\n")


async def run_query(rag: RAGService, question: str):
    print(f"\n🔍 Query: {question}")
    print("   Processing...\n")

    result = await rag.query(question)

    print("─" * 65)
    print("📋 ANSWER:\n")
    print(result.answer)
    print("\n─" * 65)
    print(f"⚡ Latency: {result.latency_ms}ms  |  🪙 Tokens: {result.tokens_used}  |  🔍 Chunks: {result.chunks_retrieved}")

    if result.citations:
        print("\n📎 SOURCES:")
        for c in result.citations:
            score_pct = int(c['relevance_score'] * 100)
            print(f"  [{c['citation_id']}] {c['source_file']} (section {c['section']}) — {score_pct}% relevant")
            print(f"      \"{c['excerpt'][:120]}...\"")
    print()


async def interactive_mode(rag: RAGService):
    print_banner()
    print("  Type your question and press Enter. Type 'quit' to exit.\n")

    chat_history = []
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        await run_query(rag, question)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', '-q', help='Run a single query and exit')
    args = parser.parse_args()

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith('sk-your'):
        print("❌ OPENAI_API_KEY not set in backend/.env")
        sys.exit(1)

    print("⏳ Initializing RAG service...")
    rag = RAGService()
    await rag.initialize()

    if not rag.is_ready:
        print("❌ RAG service failed to initialize. Run: python scripts/ingest_documents.py first")
        sys.exit(1)

    stats = rag.get_stats()
    print(f"✅ Ready — {stats['total_chunks_indexed']} chunks indexed\n")

    if args.question:
        await run_query(rag, args.question)
    else:
        await interactive_mode(rag)


if __name__ == '__main__':
    asyncio.run(main())
