from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
import uvicorn

from config import settings
from .rag.pipeline import RAGPipeline
from .models.schemas import QueryRequest, QueryResponse, HealthResponse
from .cognitive import NeuroelasticGraph, MIDASCognitiveEngine, HumanizerLLM, AphelionLayer
from .cognitive.gnn_reasoner import GNNInferenceEngine

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Microservico de IA para consultas financeiras usando RAG hibrido + FDR v2",
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

# FDR v2 Components
graph = NeuroelasticGraph()
aphelion = AphelionLayer(graph)
cognitive_engine = MIDASCognitiveEngine(graph, aphelion)
humanizer = HumanizerLLM()


@app.on_event("startup")
async def startup_event():
    await rag_pipeline.initialize()
    ROOT_DIR = Path(__file__).resolve().parent.parent 
    GNN_MODEL_FILENAME = "gnn_neuroelastic_pretrained.pt"
    GNN_MODEL_PATH = ROOT_DIR / GNN_MODEL_FILENAME # Cria o path absoluto
    
    # print para debug!
    print(f"DEBUG: Tentando carregar GNN de: {GNN_MODEL_PATH}")
    
    try:
        if not GNN_MODEL_PATH.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {GNN_MODEL_PATH}")
            
        # 1. Instancia o motor GNN
        gnn_inference_engine = GNNInferenceEngine(model_path=str(GNN_MODEL_PATH))
        
        # 2. Injeta o motor carregado no NeuroelasticGraph (objeto 'graph')
        if gnn_inference_engine.model is not None:
            graph.gnn_engine = gnn_inference_engine
            graph.use_gnn = True
            print(f"✅ GNN Engine carregado e injetado. GNN ATIVO.")
        else:
            raise ValueError("O NeuroelasticGNN foi instanciado, mas falhou ao carregar pesos ou está inválido.")

    except FileNotFoundError as fnf_e:
        print(f"❌ Erro GNN: Arquivo do modelo não encontrado. GNN DESATIVADO. Detalhe: {fnf_e}")
        graph.gnn_engine = None 
        graph.use_gnn = False
    except Exception as e:
        print(f"❌ Erro FATAL GNN (Pytorch ou Outro). Desativando GNN. Detalhe: {e}")
        graph.gnn_engine = None 
        graph.use_gnn = False


@app.on_event("shutdown")
async def shutdown_event():
    await graph.close()


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
        
        # FDR v2 health
        graph_stats = await graph.get_stats()
        checks["fdr_v2"] = {
            "graph_nodes": graph_stats["nodes"],
            "graph_edges": graph_stats["edges"],
            "coherence": graph_stats["coherence"],
            "extinctions": len(aphelion.extinction_history)
        }
            
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


# =============================================================================
# FDR v2 ENDPOINTS
# =============================================================================

class CognitiveQueryRequest(BaseModel):
    query: str
    user_id: Optional[int] = None


@app.post("/v2/query")
async def cognitive_query(request: CognitiveQueryRequest):
    """
    FDR v2 - Query cognitiva com ICE (Interface Cognitiva Estruturada)
    """
    import time
    start_time = time.time()
    
    try:
        # Raciocínio cognitivo
        ice_output = await cognitive_engine.reason(request.query)
        
        # Humanização (LLM "algemado")
        humanized_response = await humanizer.humanize(ice_output)
        ice_output.humanized_response = humanized_response
        
        duration_ms = int((time.time() - start_time) * 1000)
        ice_output.metadata["duration_ms"] = duration_ms
        
        return ice_output.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no raciocínio cognitivo: {str(e)}")


@app.post("/graph/populate")
async def populate_graph(background_tasks: BackgroundTasks):
    """
    Popula grafo com dados dummy AMPLIADOS para treinar GNN
    """
    async def _populate():
        dummy_transactions = [
        {"transaction_id": "tx001", "merchant": "ifood", "category": "alimentação", "amount": 45.0, "description": "delivery almoço hamburger", "timestamp": "2024-01-15T12:00:00"},
        {"transaction_id": "tx002", "merchant": "uber", "category": "transporte", "amount": 18.5, "description": "corrida centro trabalho", "timestamp": "2024-01-15T14:30:00"},
        {"transaction_id": "tx003", "merchant": "ifood", "category": "alimentação", "amount": 52.0, "description": "delivery jantar pizza", "timestamp": "2024-01-16T20:00:00"},
        {"transaction_id": "tx004", "merchant": "rappi", "category": "alimentação", "amount": 38.0, "description": "delivery lanche açaí", "timestamp": "2024-01-17T16:00:00"},
        {"transaction_id": "tx005", "merchant": "uber", "category": "transporte", "amount": 22.0, "description": "corrida aeroporto viagem", "timestamp": "2024-01-18T08:00:00"},
        {"transaction_id": "tx006", "merchant": "amazon", "category": "compras", "amount": 120.0, "description": "livros técnicos programação", "timestamp": "2024-01-19T10:00:00"},
        {"transaction_id": "tx007", "merchant": "ifood", "category": "alimentação", "amount": 41.0, "description": "delivery pizza calabresa", "timestamp": "2024-01-20T19:30:00"},
        {"transaction_id": "tx008", "merchant": "magazine luiza", "category": "compras", "amount": 350.0, "description": "notebook dell inspiron", "timestamp": "2024-01-21T15:00:00"},
        {"transaction_id": "tx009", "merchant": "rappi", "category": "alimentação", "amount": 29.0, "description": "delivery café starbucks", "timestamp": "2024-01-22T09:00:00"},
        {"transaction_id": "tx010", "merchant": "uber", "category": "transporte", "amount": 15.0, "description": "corrida shopping morumbi", "timestamp": "2024-01-22T18:00:00"},
        {"transaction_id": "tx011", "merchant": "ifood", "category": "alimentação", "amount": 58.0, "description": "delivery sushi japonês", "timestamp": "2024-01-23T20:00:00"},
        {"transaction_id": "tx012", "merchant": "rappi", "category": "alimentação", "amount": 42.0, "description": "delivery hamburguer artesanal", "timestamp": "2024-01-24T19:00:00"},
        {"transaction_id": "tx013", "merchant": "uber", "category": "transporte", "amount": 25.0, "description": "corrida paulista escritório", "timestamp": "2024-01-25T08:30:00"},
        {"transaction_id": "tx014", "merchant": "mercado livre", "category": "compras", "amount": 89.0, "description": "fone bluetooth jbl", "timestamp": "2024-01-26T14:00:00"},
        {"transaction_id": "tx015", "merchant": "ifood", "category": "alimentação", "amount": 36.0, "description": "delivery marmita fitness", "timestamp": "2024-01-27T12:30:00"},
        {"transaction_id": "tx016", "merchant": "netflix", "category": "entretenimento", "amount": 55.0, "description": "assinatura mensal premium", "timestamp": "2024-01-28T10:00:00"},
        {"transaction_id": "tx017", "merchant": "uber", "category": "transporte", "amount": 19.0, "description": "corrida casa noite", "timestamp": "2024-01-28T23:00:00"},
        {"transaction_id": "tx018", "merchant": "amazon", "category": "compras", "amount": 145.0, "description": "mouse gamer logitech", "timestamp": "2024-01-29T16:00:00"},
        {"transaction_id": "tx019", "merchant": "rappi", "category": "alimentação", "amount": 33.0, "description": "delivery pizza margherita", "timestamp": "2024-01-30T21:00:00"},
        {"transaction_id": "tx020", "merchant": "spotify", "category": "entretenimento", "amount": 21.0, "description": "assinatura premium individual", "timestamp": "2024-01-31T09:00:00"},
    ]
    
        for tx in dummy_transactions:
            await graph.add_transaction_node(**tx)

    
    background_tasks.add_task(_populate)
    return {"status": "populating", "message": "Grafo sendo populado com 20 transações (dataset para GNN)"}


@app.get("/graph/stats")
async def graph_stats():
    """
    Estatísticas do grafo neuroelástico + status GNN
    """
    stats = await graph.get_stats()
    extinction_summary = await aphelion.get_extinction_summary()
    
    return {
        "graph": stats,
        "aphelion": extinction_summary,
        "gnn": {
            "enabled": graph.use_gnn,
            "model_loaded": graph.gnn_engine is not None
        }
    }


@app.post("/gnn/train")
async def train_gnn_endpoint(background_tasks: BackgroundTasks):
    """
    Treina GNN com dados do grafo (background task)
    """
    import subprocess
    try:
        subprocess.run(["python", "scripts/train_gnn.py"], check=False)
    except Exception as e:
        print(f"Error at train GNN :/, see te error here: {e}")
    return {"status": "training_started", "message": "GNN training iniciado em background"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )