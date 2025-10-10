import time
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import structlog
from ..models.schemas import QueryType, RetrievalResult
from .query_classifier import QueryClassifier
from .retrievers import FAISSRetriever, TFIDFRetriever, SQLRetriever
from .postprocessor import ResponsePostprocessor
from config import settings

logger = structlog.get_logger()


class RAGPipeline:
    def __init__(self):
        self.embeddings = None
        self.query_classifier = None
        self.faiss_retriever = None
        self.tfidf_retriever = None
        self.sql_retriever = None
        self.postprocessor = None
        self.initialized = False
        self._embedding_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    async def initialize(self):
        try:
            logger.info("Inicializando RAG Pipeline...")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL
            )
            self.query_classifier = QueryClassifier()
            self.faiss_retriever = FAISSRetriever(self.embeddings)
            self.tfidf_retriever = TFIDFRetriever()
            self.sql_retriever = SQLRetriever()
            self.postprocessor = ResponsePostprocessor()
            
            await self._load_indexes()
            
            self.initialized = True
            logger.info("RAG Pipeline inicializado com sucesso!")
            
        except Exception as e:
            logger.error("Erro ao inicializar RAG Pipeline", error=str(e))
            raise
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self._embedding_cache:
            self._cache_hits += 1
            logger.debug("embedding_cache_hit", query_hash=cache_key[:8])
            return self._embedding_cache[cache_key]
        
        self._cache_misses += 1
        return None
    
    def _cache_embedding(self, text: str, embedding: List[float]):
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if len(self._embedding_cache) >= 1000:
            first_key = next(iter(self._embedding_cache))
            del self._embedding_cache[first_key]
        
        self._embedding_cache[cache_key] = embedding
        logger.debug("embedding_cached", query_hash=cache_key[:8], cache_size=len(self._embedding_cache))
    
    async def process_query(
        self, 
        query: str, 
        user_id: int, 
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        logger.info(
            "pipeline_started",
            user_id=user_id,
            query_length=len(query),
            has_filters=filters is not None
        )
        
        try:
            processed_query = await self._preprocess_query(query)
            logger.debug("query_preprocessed", processed=processed_query[:50])
            
            query_type = await self.query_classifier.classify(processed_query)
            logger.info("query_classified", type=query_type.value)
            
            retrieval_start = time.time()
            retrieval_results = await self._hybrid_retrieve(
                processed_query, user_id, query_type, filters
            )
            retrieval_time = int((time.time() - retrieval_start) * 1000)
            logger.info("retrieval_completed", duration_ms=retrieval_time, results=len(retrieval_results))
            
            final_answer = await self.postprocessor.format_response(
                query, query_type, retrieval_results
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            confidence = self._calculate_confidence(retrieval_results)
            
            logger.info(
                "pipeline_completed",
                total_duration_ms=execution_time,
                confidence=confidence,
                cache_hit_rate=self._cache_hits / max(1, self._cache_hits + self._cache_misses)
            )
            
            return {
                "answer": final_answer,
                "query_type": query_type,
                "retrieval_results": retrieval_results,
                "confidence": confidence,
                "execution_time_ms": execution_time
            }
            
        except Exception as e:
            logger.error(
                "pipeline_error",
                query=query[:100],
                user_id=user_id,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000)
            )
            raise
    
    async def _hybrid_retrieve(
        self,
        query: str,
        user_id: int,
        query_type: QueryType,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:

        all_results = []
        

        faiss_results = await self.faiss_retriever.retrieve(
            query, user_id, top_k=settings.TOP_K_RETRIEVAL
        )
        all_results.extend(faiss_results)
        
    
        tfidf_results = await self.tfidf_retriever.retrieve(
            query, user_id, top_k=settings.TOP_K_RETRIEVAL
        )
        all_results.extend(tfidf_results)
        

        if query_type in [QueryType.BALANCE, QueryType.SUBSCRIPTIONS]:
            sql_results = await self.sql_retriever.retrieve(
                query, user_id, query_type, filters
            )
            all_results.extend(sql_results)
        
 
        ranked_results = self._hybrid_ranking(all_results)
        
        return ranked_results[:settings.TOP_K_RETRIEVAL]
    
    def _hybrid_ranking(self, results: List[RetrievalResult]) -> List[RetrievalResult]:


        faiss_results = [r for r in results if r.source == "faiss"]
        tfidf_results = [r for r in results if r.source == "tfidf"]
        sql_results = [r for r in results if r.source == "sql"]
        
   
        for result in faiss_results:
            result.score *= settings.HYBRID_WEIGHT_FAISS
        
        for result in tfidf_results:
            result.score *= settings.HYBRID_WEIGHT_TFIDF
        

        for result in sql_results:
            result.score *= 1.2
        

        all_results = faiss_results + tfidf_results + sql_results
        all_results.sort(key=lambda x: x.score, reverse=True)
        
    
        deduplicated = self._remove_duplicates(all_results)
        
        return deduplicated
    
    def _remove_duplicates(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        if not results:
            return []
        if len(results) <= 1:
            return results
        
        unique_results = [results[0]]
        
        for result in results[1:]:
            is_duplicate = False
            for unique_result in unique_results:
                similarity = self._content_similarity(result.content, unique_result.content)
                if similarity > 0.8:  
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
        
        return unique_results
    
    def _content_similarity(self, content1: str, content2: str) -> float:
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_confidence(self, results: List[RetrievalResult]) -> float:
        if not results:
            return 0.0
        
        avg_score = sum(r.score for r in results) / len(results)
        return min(avg_score, 1.0)
    
    async def _preprocess_query(self, query: str) -> str:
        processed = query.strip().lower()
        
        # Financial terms normalization
        processed = processed.replace("r$", "reais")
        processed = processed.replace("$", "reais")
        
        return processed
    
    async def _load_indexes(self):
        try:
            await self.faiss_retriever.load_index()
            await self.tfidf_retriever.load_index()
            logger.info("Índices carregados com sucesso")
        except Exception as e:
            logger.warning("Erro ao carregar índices", error=str(e))
    
    async def rebuild_user_embeddings(self, user_id: int):
     ...
    
    async def get_embeddings_status(self) -> Dict[str, Any]:
        return {
            "faiss_initialized": self.faiss_retriever.is_initialized() if self.faiss_retriever else False,
            "tfidf_initialized": self.tfidf_retriever.is_initialized() if self.tfidf_retriever else False,
            "pipeline_ready": self.initialized,
            "last_updated": datetime.now().isoformat()
        }