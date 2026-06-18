import os
import logging
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from openai import AsyncOpenAI
import faiss
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    metadata: Dict[str, Any]


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


@dataclass
class RAGResponse:
    answer: str
    citations: List[Dict[str, Any]]
    query: str
    latency_ms: float
    tokens_used: int
    chunks_retrieved: int


class DocumentLoader:
    """Loads and chunks documents from the data directory."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self, docs_path: str) -> List[DocumentChunk]:
        chunks = []
        docs_dir = Path(docs_path)

        if not docs_dir.exists():
            logger.warning(f"Documents directory not found: {docs_path}")
            return chunks

        for file_path in docs_dir.glob("*.txt"):
            logger.info(f"Loading document: {file_path.name}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                file_chunks = self._chunk_text(text, file_path.name)
                chunks.extend(file_chunks)
                logger.info(f"  → {len(file_chunks)} chunks created from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Total chunks loaded: {len(chunks)}")
        return chunks

    def _chunk_text(self, text: str, source_file: str) -> List[DocumentChunk]:
        chunks = []
        # Split on double newlines first to respect document sections
        sections = text.split("\n\n")
        
        current_chunk = ""
        chunk_index = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # If adding this section would exceed chunk size, save current and start new
            if len(current_chunk) + len(section) + 2 > self.chunk_size and current_chunk:
                if len(current_chunk.strip()) > 50:  # Minimum chunk size
                    chunks.append(DocumentChunk(
                        chunk_id=f"{source_file}_{chunk_index}",
                        text=current_chunk.strip(),
                        source_file=source_file,
                        page_number=chunk_index + 1,
                        metadata={
                            "source": source_file,
                            "chunk_index": chunk_index,
                            "char_count": len(current_chunk),
                        }
                    ))
                    chunk_index += 1
                    # Overlap: keep last portion of current chunk
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + section
                else:
                    current_chunk += "\n\n" + section
            else:
                current_chunk += ("\n\n" if current_chunk else "") + section

        # Save last chunk
        if current_chunk.strip() and len(current_chunk.strip()) > 50:
            chunks.append(DocumentChunk(
                chunk_id=f"{source_file}_{chunk_index}",
                text=current_chunk.strip(),
                source_file=source_file,
                page_number=chunk_index + 1,
                metadata={
                    "source": source_file,
                    "chunk_index": chunk_index,
                    "char_count": len(current_chunk),
                }
            ))

        return chunks


class EmbeddingService:
    """Generates embeddings using OpenAI text-embedding-3-small."""

    def __init__(self, client: AsyncOpenAI, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model

    async def embed_texts(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Embedding batch {i // batch_size + 1} ({len(batch)} texts)...")
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    async def embed_query(self, query: str) -> np.ndarray:
        response = await self.client.embeddings.create(
            model=self.model,
            input=[query],
        )
        return np.array(response.data[0].embedding, dtype=np.float32)


class FAISSVectorStore:
    """FAISS-based vector store for similarity search."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
        self.chunks: List[DocumentChunk] = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[DocumentChunk]):
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} vectors to FAISS index. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievedChunk]:
        query_embedding = query_embedding.reshape(1, -1).copy()
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievedChunk(
                chunk=self.chunks[idx],
                score=float(score),
            ))

        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"FAISS index saved to {path}")

    def load(self, path: str) -> bool:
        index_path = os.path.join(path, "index.faiss")
        chunks_path = os.path.join(path, "chunks.pkl")
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            logger.info(f"FAISS index loaded from {path}. Vectors: {self.index.ntotal}")
            return True
        return False


class RAGService:
    """Main RAG orchestration service."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.loader = DocumentLoader(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.embedding_service = EmbeddingService(
            client=self.client,
            model=settings.EMBEDDING_MODEL,
        )
        self.vector_store = FAISSVectorStore(dimension=1536)
        self.is_ready = False
        self.total_documents = 0
        self.total_chunks = 0

    async def initialize(self):
        """Load documents and build the FAISS index."""
        # Try loading existing index first
        if self.vector_store.load(settings.FAISS_INDEX_PATH):
            self.is_ready = True
            self.total_chunks = self.vector_store.index.ntotal
            logger.info("Loaded existing FAISS index.")
            return

        # Otherwise build from scratch
        chunks = self.loader.load_documents(settings.DOCS_PATH)
        if not chunks:
            logger.warning("No documents found to index.")
            return

        self.total_chunks = len(chunks)
        self.total_documents = len(set(c.source_file for c in chunks))

        texts = [c.text for c in chunks]
        embeddings = await self.embedding_service.embed_texts(texts)

        self.vector_store.add_embeddings(embeddings, chunks)
        self.vector_store.save(settings.FAISS_INDEX_PATH)
        self.is_ready = True
        logger.info(f"RAG service initialized: {self.total_documents} docs, {self.total_chunks} chunks.")

    async def query(
        self,
        question: str,
        top_k: int = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> RAGResponse:
        if not self.is_ready:
            raise RuntimeError("RAG service not initialized. Please wait for indexing to complete.")

        start_time = time.time()
        top_k = top_k or settings.TOP_K_RETRIEVAL

        # Step 1: Embed the query
        query_embedding = await self.embedding_service.embed_query(question)

        # Step 2: Retrieve relevant chunks
        retrieved = self.vector_store.search(query_embedding, top_k=top_k)

        # Filter by similarity threshold
        retrieved = [r for r in retrieved if r.score >= settings.SIMILARITY_THRESHOLD]

        if not retrieved:
            return RAGResponse(
                answer="I could not find relevant information in the Ascension Via Christi knowledge base to answer this question. Please rephrase your query or contact the relevant department directly.",
                citations=[],
                query=question,
                latency_ms=round((time.time() - start_time) * 1000, 2),
                tokens_used=0,
                chunks_retrieved=0,
            )

        # Step 3: Build context
        context_parts = []
        citations = []
        for i, result in enumerate(retrieved, 1):
            context_parts.append(
                f"[Source {i}: {result.chunk.source_file} | Section {result.chunk.page_number} | Relevance: {result.score:.2f}]\n"
                f"{result.chunk.text}"
            )
            citations.append({
                "citation_id": i,
                "source_file": result.chunk.source_file,
                "section": result.chunk.page_number,
                "relevance_score": round(result.score, 4),
                "excerpt": result.chunk.text[:300] + "..." if len(result.chunk.text) > 300 else result.chunk.text,
            })

        context = "\n\n---\n\n".join(context_parts)

        # Step 4: Build messages
        system_prompt = """You are the Ascension Via Christi Health AI Knowledge Assistant — a trusted clinical and operational reference system.

Your role is to provide accurate, citation-grounded answers based ONLY on the provided context documents.

Guidelines:
- Answer clearly and concisely using only the information in the provided context
- Always cite your sources using [Source N] notation inline
- If the context doesn't fully answer the question, say so explicitly
- For clinical questions, always note that clinical judgment and direct consultation with the care team takes precedence
- Format your response with clear structure when answering multi-part questions
- Never fabricate information not present in the context

Ascension Via Christi Health | HIPAA Compliant System | For authorized personnel only"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history[-6:])  # Last 3 turns

        messages.append({
            "role": "user",
            "content": f"Context from Ascension knowledge base:\n\n{context}\n\n---\n\nQuestion: {question}"
        })

        # Step 5: Generate answer
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return RAGResponse(
            answer=answer,
            citations=citations,
            query=question,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            chunks_retrieved=len(retrieved),
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "total_chunks_indexed": self.vector_store.index.ntotal if self.is_ready else 0,
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.OPENAI_MODEL,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k": settings.TOP_K_RETRIEVAL,
        }
