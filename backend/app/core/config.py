from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # RAG Configuration
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.3

    # Document storage
    DOCS_PATH: str = "data/documents"
    FAISS_INDEX_PATH: str = "data/faiss_index"

    # Redis (optional caching)
    REDIS_URL: Optional[str] = None

    # App
    APP_NAME: str = "Ascension Via Christi RAG System"
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
