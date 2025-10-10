import os
import pickle
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import structlog

from ..models.schemas import RetrievalResult, QueryType
from config import settings

logger = structlog.get_logger()


class BaseRetriever(ABC):    
    @abstractmethod
    async def retrieve(self, query: str, user_id: int, **kwargs) -> List[RetrievalResult]:
        pass


class FAISSRetriever(BaseRetriever):
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.vector_store = None
        self.documents = []
        
    async def retrieve(self, query: str, user_id: int, top_k: int = 10) -> List[RetrievalResult]:
        if not self.vector_store:
            return []
        
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=top_k
            )
            
            results = []
            for doc, score in docs_with_scores:
                if "user_id" in doc.metadata and doc.metadata["user_id"] != user_id:
                    continue
                
            
                similarity = 1 / (1 + score)
                
                result = RetrievalResult(
                    content=doc.page_content,
                    score=similarity,
                    source="faiss",
                    metadata=doc.metadata
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error("Erro no FAISS retriever", error=str(e))
            return []
    
    async def load_index(self):
        try:
            if os.path.exists(settings.FAISS_INDEX_PATH):
                self.vector_store = FAISS.load_local(
                    settings.FAISS_INDEX_PATH, 
                    self.embeddings
                )
                logger.info("Índice FAISS carregado")
        except Exception as e:
            logger.warning("Erro ao carregar índice FAISS", error=str(e))
    
    def is_initialized(self) -> bool:
        return self.vector_store is not None


class TFIDFRetriever(BaseRetriever):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words= None, # poderia adicionar stopwords em pt-br
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.documents = []
        self.document_metadata = []
    
    async def retrieve(self, query: str, user_id: int, top_k: int = 10) -> List[RetrievalResult]:
        if self.tfidf_matrix is None:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            
            
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] < 0.1:  
                    continue
                
            
                metadata = self.document_metadata[idx] if idx < len(self.document_metadata) else {}
                if "user_id" in metadata and metadata["user_id"] != user_id:
                    continue
                
                result = RetrievalResult(
                    content=self.documents[idx],
                    score=float(similarities[idx]),
                    source="tfidf",
                    metadata=metadata
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error("Erro no TF-IDF retriever", error=str(e))
            return []
    
    async def load_index(self):
        try:
            vectorizer_path = os.path.join(settings.TFIDF_INDEX_PATH, "vectorizer.pkl")
            matrix_path = os.path.join(settings.TFIDF_INDEX_PATH, "matrix.pkl")
            docs_path = os.path.join(settings.TFIDF_INDEX_PATH, "documents.pkl")
            
            if all(os.path.exists(p) for p in [vectorizer_path, matrix_path, docs_path]):
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(matrix_path, 'rb') as f:
                    self.tfidf_matrix = pickle.load(f)
                with open(docs_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", [])
                    self.document_metadata = data.get("metadata", [])
                
                logger.info("Índice TF-IDF carregado")
        except Exception as e:
            logger.warning("Erro ao carregar índice TF-IDF", error=str(e))
    
    def is_initialized(self) -> bool:
        return self.tfidf_matrix is not None


class SQLRetriever(BaseRetriever):
    def __init__(self):
        # irei inicializar conexao com banco de dados aqui, sendo oracle "dona da verdade"
        self.db_connection = None
    
    async def retrieve(
        self, 
        query: str, 
        user_id: int, 
        query_type: QueryType,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        # a ideia é adicionar busca estruturda com sql
        
        results = []
        
        if query_type == QueryType.BALANCE:
            # Mockando consulta de saldo
            result = RetrievalResult(
                content=f"Saldo atual da conta: R$ 1.250,30",
                score=1.0,
                source="sql",
                metadata={
                    "query_type": "balance",
                    "table": "accounts",
                    "user_id": user_id
                }
            )
            results.append(result)
        
        elif query_type == QueryType.SUBSCRIPTIONS:
            subscriptions = [
                "Netflix - R$ 25,90/mês",
                "Spotify - R$ 16,90/mês", 
                "Amazon Prime - R$ 9,90/mês"
            ]
            
            for sub in subscriptions:
                result = RetrievalResult(
                    content=f"Assinatura ativa: {sub}",
                    score=0.95,
                    source="sql",
                    metadata={
                        "query_type": "subscriptions",
                        "table": "subscriptions",
                        "user_id": user_id
                    }
                )
                results.append(result)
        
        return results
    
    def is_initialized(self) -> bool:
        return True