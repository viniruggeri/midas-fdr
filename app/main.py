from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from config import settings
from .rag.pipeline import RAGPipeline
from .models.schemas import QueryRequest, QueryResponse, HealthResponse

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Microservico de IA para consultas financeiras usando RAG hibrido",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline = RAGPipeline()


@app.on_event("startup")
async def startup_event():
    await rag_pipeline.initialize()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    checks = {}
    status = "healthy"
    
    try:
        if rag_pipeline.initialized:
            checks["pipeline_initialized"] = True
            checks["faiss_loaded"] = rag_pipeline.faiss_retriever is not None
            checks["tfidf_loaded"] = rag_pipeline.tfidf_retriever is not None
            checks["sql_loaded"] = rag_pipeline.sql_retriever is not None
            checks["embeddings_model"] = settings.EMBEDDING_MODEL
            checks["cache_size"] = len(rag_pipeline._embedding_cache)
            checks["cache_hits"] = rag_pipeline._cache_hits
            checks["cache_misses"] = rag_pipeline._cache_misses
            
            total_requests = rag_pipeline._cache_hits + rag_pipeline._cache_misses
            if total_requests > 0:
                checks["cache_hit_rate"] = round(rag_pipeline._cache_hits / total_requests, 2)
        else:
            status = "initializing"
            checks["pipeline_initialized"] = False
            
    except Exception as e:
        status = "unhealthy"
        checks["error"] = str(e)
    
    return HealthResponse(
        status=status,
        service=settings.APP_NAME,
        version=settings.VERSION,
        checks=checks
    )


@app.post("/query", response_model=QueryResponse)
async def financial_query(request: QueryRequest):
    import time
    import structlog
    
    logger = structlog.get_logger()
    start_time = time.time()
    
    try:
        logger.info(
            "query_started",
            user_id=request.user_id,
            query_preview=request.query[:50] + "..." if len(request.query) > 50 else request.query,
            filters=request.filters
        )
        
        result = await rag_pipeline.process_query(
            query=request.query,
            user_id=request.user_id,
            filters=request.filters
        )
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "query_completed",
            user_id=request.user_id,
            query_type=result.get("query_type"),
            confidence=result.get("confidence"),
            results_count=len(result.get("retrieval_results", [])),
            duration_ms=duration_ms
        )
        
        return QueryResponse(**result)
        
    except ValueError as e:
        logger.warning("validation_error", user_id=request.user_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "query_error",
            user_id=request.user_id,
            query=request.query[:100],
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Erro ao processar consulta: {str(e)}")


@app.get("/embeddings/status")
async def embeddings_status():
    return await rag_pipeline.get_embeddings_status()


@app.post("/embeddings/rebuild")
async def rebuild_embeddings(user_id: int):
    try:
        await rag_pipeline.rebuild_user_embeddings(user_id)
        return {"status": "success", "message": f"Embeddings rebuilt for user {user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )