from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api.routes import router
from app.core.config import settings
from app.services.rag_service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

rag_service_instance: RAGService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service_instance
    logger.info("Initializing RAG service and loading documents...")
    rag_service_instance = RAGService()
    await rag_service_instance.initialize()
    app.state.rag_service = rag_service_instance
    logger.info("RAG service ready.")
    yield
    logger.info("Shutting down RAG service...")


app = FastAPI(
    title="Ascension Via Christi — Enterprise RAG API",
    description="Production-grade Retrieval-Augmented Generation system for clinical knowledge retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Ascension RAG API",
        "version": "1.0.0",
    }
