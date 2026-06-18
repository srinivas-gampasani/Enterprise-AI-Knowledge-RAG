from fastapi import APIRouter, HTTPException, Request, Depends
from app.models.schemas import (
    QueryRequest, QueryResponse, CitationItem,
    IndexStatsResponse, ErrorResponse
)
from app.services.rag_service import RAGService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG knowledge base",
    description="Submit a clinical or operational question and receive a grounded answer with citations.",
)
async def query_knowledge_base(
    body: QueryRequest,
    rag: RAGService = Depends(get_rag_service),
):
    if not rag.is_ready:
        raise HTTPException(status_code=503, detail="RAG service is still initializing. Please try again in a moment.")

    try:
        result = await rag.query(
            question=body.question,
            top_k=body.top_k,
            chat_history=body.chat_history,
        )

        citations = [CitationItem(**c) for c in result.citations]

        return QueryResponse(
            answer=result.answer,
            citations=citations,
            query=result.query,
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
            chunks_retrieved=result.chunks_retrieved,
        )

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/stats",
    response_model=IndexStatsResponse,
    summary="Get RAG system statistics",
)
async def get_stats(rag: RAGService = Depends(get_rag_service)):
    stats = rag.get_stats()
    return IndexStatsResponse(**stats)


@router.get(
    "/ready",
    summary="Check if the RAG system is ready",
)
async def readiness_check(rag: RAGService = Depends(get_rag_service)):
    if rag.is_ready:
        return {"ready": True, "message": "RAG system is ready to accept queries."}
    return {"ready": False, "message": "RAG system is still initializing..."}
