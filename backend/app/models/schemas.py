from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="The clinical or operational question to ask")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of document chunks to retrieve")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Previous conversation turns")


class CitationItem(BaseModel):
    citation_id: int
    source_file: str
    section: int
    relevance_score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationItem]
    query: str
    latency_ms: float
    tokens_used: int
    chunks_retrieved: int
    status: str = "success"


class IndexStatsResponse(BaseModel):
    is_ready: bool
    total_chunks_indexed: int
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    status: str = "success"


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    detail: Optional[str] = None


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text content to ingest")
    source_name: str = Field(..., description="Name/identifier for this document")
    metadata: Optional[Dict[str, Any]] = Field(default=None)
