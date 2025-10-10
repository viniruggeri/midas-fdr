import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Midas AI Service"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database PostgreSQL (Vector DB)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "midas_vectors"
    POSTGRES_USER: str = "midas"
    POSTGRES_PASSWORD: str = "midas123"
    
    # Oracle Database
    ORACLE_HOST: str = "localhost"
    ORACLE_PORT: int = 1521
    ORACLE_SERVICE: str = "XEPDB1"
    ORACLE_USER: str = "midas"
    ORACLE_PASSWORD: str = "midas123"
    
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "midas-ai-service"
    

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    TFIDF_INDEX_PATH: str = "./data/tfidf_index"
    
    
    TOP_K_RETRIEVAL: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    HYBRID_WEIGHT_FAISS: float = 0.6
    HYBRID_WEIGHT_TFIDF: float = 0.4
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()