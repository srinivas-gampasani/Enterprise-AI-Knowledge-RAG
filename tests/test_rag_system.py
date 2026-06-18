"""
Ascension Via Christi RAG System — Test Suite
Run with: pytest tests/ -v
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.rag_service import (
    DocumentLoader, DocumentChunk, FAISSVectorStore, RetrievedChunk
)


# ── DocumentLoader Tests ──────────────────────────────────
class TestDocumentLoader:

    def setup_method(self):
        self.loader = DocumentLoader(chunk_size=500, chunk_overlap=100)

    def test_chunk_text_basic(self):
        text = "Section 1\n\nThis is the first section content that has enough text to be meaningful.\n\nSection 2\n\nThis is the second section with more content for testing purposes."
        chunks = self.loader._chunk_text(text, "test.txt")
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)
            assert chunk.source_file == "test.txt"
            assert len(chunk.text) > 50

    def test_chunk_has_metadata(self):
        text = "Header\n\n" + ("Word " * 200)
        chunks = self.loader._chunk_text(text, "doc.txt")
        for c in chunks:
            assert "source" in c.metadata
            assert "chunk_index" in c.metadata

    def test_empty_text_returns_no_chunks(self):
        chunks = self.loader._chunk_text("", "empty.txt")
        assert len(chunks) == 0

    def test_short_text_returns_one_chunk(self):
        text = "This is a short document with enough content to form a single chunk and nothing more here."
        chunks = self.loader._chunk_text(text, "short.txt")
        assert len(chunks) == 1

    def test_chunk_ids_are_unique(self):
        text = "\n\n".join([f"Section {i}\n" + "Content word " * 100 for i in range(10)])
        chunks = self.loader._chunk_text(text, "multi.txt")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_load_documents_nonexistent_path(self):
        chunks = self.loader.load_documents("/nonexistent/path")
        assert chunks == []

    def test_load_documents_real_files(self, tmp_path):
        doc = tmp_path / "test_clinical.txt"
        doc.write_text("SEPSIS PROTOCOL\n\nAdminister antibiotics within 1 hour of sepsis recognition.\n\nFALL PREVENTION\n\nAssess all patients using Morse Fall Scale on admission and every shift.")
        chunks = self.loader.load_documents(str(tmp_path))
        assert len(chunks) > 0
        assert all(c.source_file == "test_clinical.txt" for c in chunks)


# ── FAISSVectorStore Tests ────────────────────────────────
class TestFAISSVectorStore:

    def setup_method(self):
        self.store = FAISSVectorStore(dimension=128)

    def _make_chunk(self, i: int) -> DocumentChunk:
        return DocumentChunk(
            chunk_id=f"test_{i}",
            text=f"Test chunk content number {i} with some meaningful text.",
            source_file="test.txt",
            page_number=i,
            metadata={"chunk_index": i}
        )

    def _make_embeddings(self, n: int, dim: int = 128) -> np.ndarray:
        np.random.seed(42)
        emb = np.random.randn(n, dim).astype(np.float32)
        return emb

    def test_add_and_search(self):
        chunks = [self._make_chunk(i) for i in range(10)]
        embeddings = self._make_embeddings(10)
        self.store.add_embeddings(embeddings, chunks)

        assert self.store.index.ntotal == 10

        query = self._make_embeddings(1)[0]
        results = self.store.search(query, top_k=3)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, RetrievedChunk)
            assert 0 <= r.score <= 1.0

    def test_search_returns_top_k(self):
        chunks = [self._make_chunk(i) for i in range(20)]
        embeddings = self._make_embeddings(20)
        self.store.add_embeddings(embeddings, chunks)

        query = self._make_embeddings(1)[0]
        results = self.store.search(query, top_k=5)
        assert len(results) == 5

    def test_scores_are_sorted_descending(self):
        chunks = [self._make_chunk(i) for i in range(15)]
        embeddings = self._make_embeddings(15)
        self.store.add_embeddings(embeddings, chunks)

        query = self._make_embeddings(1)[0]
        results = self.store.search(query, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results must be sorted by score desc"

    def test_save_and_load(self, tmp_path):
        chunks = [self._make_chunk(i) for i in range(5)]
        embeddings = self._make_embeddings(5)
        self.store.add_embeddings(embeddings, chunks)
        self.store.save(str(tmp_path))

        new_store = FAISSVectorStore(dimension=128)
        loaded = new_store.load(str(tmp_path))
        assert loaded is True
        assert new_store.index.ntotal == 5
        assert len(new_store.chunks) == 5

    def test_load_nonexistent_returns_false(self):
        result = self.store.load("/nonexistent/index/path")
        assert result is False

    def test_empty_store_search(self):
        query = self._make_embeddings(1)[0]
        results = self.store.search(query, top_k=5)
        assert results == []


# ── RAGService Integration Tests (mocked OpenAI) ──────────
class TestRAGServiceMocked:

    @pytest.mark.asyncio
    async def test_query_returns_rag_response(self, tmp_path):
        """Test full RAG pipeline with mocked OpenAI calls."""
        from app.services.rag_service import RAGService
        from app.core.config import settings

        # Write test documents
        doc_dir = tmp_path / "documents"
        doc_dir.mkdir()
        (doc_dir / "sepsis.txt").write_text(
            "SEPSIS PROTOCOL\n\nThe sepsis 3-hour bundle includes:\n"
            "1. Measure lactate level\n2. Obtain blood cultures\n"
            "3. Administer broad-spectrum antibiotics\n4. Administer 30 mL/kg crystalloid"
        )

        index_dir = tmp_path / "index"

        with patch.object(settings, 'DOCS_PATH', str(doc_dir)), \
             patch.object(settings, 'FAISS_INDEX_PATH', str(index_dir)), \
             patch.object(settings, 'OPENAI_API_KEY', 'sk-test-key'):

            service = RAGService()

            # Mock embedding calls
            mock_emb_data = MagicMock()
            mock_emb_data.embedding = [0.1] * 1536

            mock_emb_response = MagicMock()
            mock_emb_response.data = [mock_emb_data]

            service.embedding_service.client = AsyncMock()
            service.embedding_service.client.embeddings.create = AsyncMock(return_value=mock_emb_response)

            await service.initialize()
            assert service.is_ready is True
            assert service.vector_store.index.ntotal > 0

            # Mock chat completion
            mock_choice = MagicMock()
            mock_choice.message.content = "The sepsis 3-hour bundle includes measuring lactate [Source 1], obtaining blood cultures, and administering antibiotics."

            mock_usage = MagicMock()
            mock_usage.total_tokens = 150

            mock_completion = MagicMock()
            mock_completion.choices = [mock_choice]
            mock_completion.usage = mock_usage

            service.client = AsyncMock()
            service.client.chat.completions.create = AsyncMock(return_value=mock_completion)
            service.embedding_service.client = service.client

            # Re-mock embed_query
            async def mock_embed_query(text):
                return np.array([0.1] * 1536, dtype=np.float32)

            service.embedding_service.embed_query = mock_embed_query

            result = await service.query("What is the sepsis 3-hour bundle?")

            assert result.answer != ""
            assert result.query == "What is the sepsis 3-hour bundle?"
            assert result.latency_ms > 0
            assert result.tokens_used == 150


# ── API Route Tests (FastAPI TestClient) ──────────────────
class TestAPIRoutes:

    def test_health_endpoint(self):
        """Test that the /health endpoint responds correctly."""
        from fastapi.testclient import TestClient
        from app.main import app

        # Minimal mock setup
        mock_rag = MagicMock()
        mock_rag.is_ready = True
        mock_rag.get_stats.return_value = {
            "is_ready": True,
            "total_chunks_indexed": 42,
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-3.5-turbo",
            "chunk_size": 800,
            "chunk_overlap": 150,
            "top_k": 5,
        }
        app.state.rag_service = mock_rag

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")
        # Just verify it doesn't 500
        assert response.status_code in [200, 503]

    def test_stats_schema(self):
        """Test IndexStatsResponse schema validation."""
        from app.models.schemas import IndexStatsResponse
        stats = IndexStatsResponse(
            is_ready=True,
            total_chunks_indexed=100,
            embedding_model="text-embedding-3-small",
            llm_model="gpt-3.5-turbo",
            chunk_size=800,
            chunk_overlap=150,
            top_k=5,
        )
        assert stats.is_ready is True
        assert stats.total_chunks_indexed == 100

    def test_query_request_schema(self):
        """Test QueryRequest validation."""
        from app.models.schemas import QueryRequest
        req = QueryRequest(question="What is the sepsis protocol?", top_k=3)
        assert req.question == "What is the sepsis protocol?"
        assert req.top_k == 3

    def test_query_request_min_length(self):
        """Test that short questions are rejected."""
        from app.models.schemas import QueryRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryRequest(question="Hi")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
