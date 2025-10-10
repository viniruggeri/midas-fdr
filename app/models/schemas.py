from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    SPENDING = "spending"
    SUBSCRIPTIONS = "subscriptions" 
    BALANCE = "balance"
    CATEGORIES = "categories"
    GOALS = "goals"
    GENERAL = "general"


class QueryRequest(BaseModel):
    query: str = Field(..., description="Pergunta em linguagem natural")
    user_id: int = Field(..., description="ID do usuário")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filtros adicionais")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError("Query muito curta. Mínimo de 3 caracteres.")
        if len(v) > 500:
            raise ValueError("Query muito longa. Máximo de 500 caracteres.")
        return v.strip()
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if v <= 0:
            raise ValueError("user_id deve ser maior que zero")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Quanto gastei com delivery este mês?",
                "user_id": 123,
                "filters": {
                    "month": "2025-10",
                    "category": "delivery"
                }
            }
        }


class RetrievalResult(BaseModel):
    content: str = Field(..., description="Conteúdo encontrado")
    score: float = Field(..., description="Score de relevância")
    source: str = Field(..., description="Fonte: faiss, tfidf ou sql")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Resposta em linguagem natural")
    query_type: QueryType = Field(..., description="Tipo de consulta identificado")
    retrieval_results: List[RetrievalResult] = Field(..., description="Resultados da busca RAG")
    confidence: float = Field(..., description="Confiança na resposta (0-1)")
    execution_time_ms: int = Field(..., description="Tempo de execução em ms")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Você gastou R$ 245,30 com delivery este mês, sendo R$ 120 no iFood e R$ 125,30 no Uber Eats.",
                "query_type": "spending",
                "retrieval_results": [
                    {
                        "content": "Transação: iFood - R$ 35,90 - 2025-10-15",
                        "score": 0.89,
                        "source": "faiss",
                        "metadata": {"category": "delivery", "bank": "nubank"}
                    }
                ],
                "confidence": 0.92,
                "execution_time_ms": 450
            }
        }


class HealthResponse(BaseModel):
    status: str = Field(..., description="Status do serviço")
    service: str = Field(..., description="Nome do serviço")
    version: str = Field(..., description="Versão")
    timestamp: datetime = Field(default_factory=datetime.now)
    checks: Optional[Dict[str, Any]] = Field(None, description="Verificações detalhadas")


class EmbeddingStatus(BaseModel):
    faiss_index_size: int
    tfidf_index_size: int
    last_updated: Optional[datetime]
    total_documents: int