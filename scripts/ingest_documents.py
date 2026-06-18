#!/usr/bin/env python3
"""
Ascension Via Christi RAG System — Document Ingestion Script

Usage:
    python scripts/ingest_documents.py                        # Re-index all docs in data/documents/
    python scripts/ingest_documents.py --file path/to/doc.txt # Add a single document
    python scripts/ingest_documents.py --rebuild              # Force rebuild (ignore cached index)

Run from the backend/ directory:
    cd backend && python ../scripts/ingest_documents.py
"""
import asyncio
import argparse
import shutil
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))

from app.services.rag_service import DocumentLoader, EmbeddingService, FAISSVectorStore
from app.core.config import settings
from openai import AsyncOpenAI


async def ingest_all(rebuild: bool = False):
    if rebuild and os.path.exists(settings.FAISS_INDEX_PATH):
        print(f"🗑  Removing existing index at {settings.FAISS_INDEX_PATH}...")
        shutil.rmtree(settings.FAISS_INDEX_PATH)

    print(f"\n📂 Loading documents from: {settings.DOCS_PATH}")
    loader = DocumentLoader(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    chunks = loader.load_documents(settings.DOCS_PATH)

    if not chunks:
        print("❌ No documents found. Add .txt files to backend/data/documents/")
        return

    print(f"✅ Loaded {len(chunks)} chunks from {len(set(c.source_file for c in chunks))} documents")

    print("\n🔢 Generating embeddings via OpenAI...")
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    emb_service = EmbeddingService(client, settings.EMBEDDING_MODEL)

    t0 = time.time()
    texts = [c.text for c in chunks]
    embeddings = await emb_service.embed_texts(texts)
    elapsed = time.time() - t0

    print(f"✅ Embeddings generated in {elapsed:.1f}s — shape: {embeddings.shape}")

    print("\n📦 Building FAISS index...")
    store = FAISSVectorStore(dimension=embeddings.shape[1])
    store.add_embeddings(embeddings, chunks)
    store.save(settings.FAISS_INDEX_PATH)

    print(f"\n🎉 Done! Index saved to {settings.FAISS_INDEX_PATH}")
    print(f"   Vectors in index: {store.index.ntotal}")
    print(f"   Start the API with: cd backend && uvicorn app.main:app --reload --port 8000")


async def ingest_single(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    # Copy file to documents directory
    dest = os.path.join(settings.DOCS_PATH, os.path.basename(file_path))
    shutil.copy2(file_path, dest)
    print(f"📋 Copied {file_path} → {dest}")

    print("🔄 Rebuilding index with new document...")
    await ingest_all(rebuild=True)


def main():
    parser = argparse.ArgumentParser(description='Ingest documents into the Ascension RAG system')
    parser.add_argument('--file', help='Path to a single document to add')
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild of entire index')
    args = parser.parse_args()

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.startswith('sk-your'):
        print("❌ OPENAI_API_KEY not set. Copy backend/.env.example to backend/.env and add your key.")
        sys.exit(1)

    print("=" * 60)
    print("  Ascension Via Christi — RAG Document Ingestion")
    print("=" * 60)

    if args.file:
        asyncio.run(ingest_single(args.file))
    else:
        asyncio.run(ingest_all(rebuild=args.rebuild))


if __name__ == '__main__':
    main()
