import json
import os
from typing import List, Dict, Any
from datetime import datetime
import pickle

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from config import settings


class DummyDataLoader:
    
    def __init__(self):
        self.data_path = "./data/dummy"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL
        )
    
    def load_all_data(self) -> Dict[str, Any]:
        data = {}
        
     
        files = [
            "transactions.json",
            "subscriptions.json", 
            "accounts.json",
            "goals.json",
            "documents_for_rag.json"
        ]
        
        for file in files:
            file_path = os.path.join(self.data_path, file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    key = file.replace('.json', '')
                    data[key] = json.load(f)
        
        return data
    
    def create_faiss_index(self) -> FAISS:
        print( "Criando índice FAISS com dados dummy...")
        

        with open(os.path.join(self.data_path, "documents_for_rag.json"), 'r', encoding='utf-8') as f:
            rag_data = json.load(f)
        
        documents = []
        for item in rag_data:
            doc = Document(
                page_content=item["text"],
                metadata=item["metadata"]
            )
            documents.append(doc)
        
    
        if documents:
            vector_store = FAISS.from_documents(documents, self.embeddings)
            
            
            os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
            vector_store.save_local(settings.FAISS_INDEX_PATH)
            
            print(f"Índice FAISS criado com {len(documents)} documentos")
            return vector_store
        
        return None
    
    def create_tfidf_index(self):
        print("Criando índice TF-IDF com dados dummy...")
        
        with open(os.path.join(self.data_path, "documents_for_rag.json"), 'r', encoding='utf-8') as f:
            rag_data = json.load(f)
        
        texts = [item["text"] for item in rag_data]
        metadata = [item["metadata"] for item in rag_data]
        
        
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words=None  
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
    
        os.makedirs(settings.TFIDF_INDEX_PATH, exist_ok=True)
        
        with open(os.path.join(settings.TFIDF_INDEX_PATH, "vectorizer.pkl"), 'wb') as f:
            pickle.dump(vectorizer, f)
        
        with open(os.path.join(settings.TFIDF_INDEX_PATH, "matrix.pkl"), 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        
        with open(os.path.join(settings.TFIDF_INDEX_PATH, "documents.pkl"), 'wb') as f:
            pickle.dump({
                "documents": texts,
                "metadata": metadata
            }, f)
        
        print(f"Índice TF-IDF criado com {len(texts)} documentos")
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        data = self.load_all_data()
        
        stats = {
            "total_transactions": len(data.get("transactions", [])),
            "total_subscriptions": len(data.get("subscriptions", [])),
            "total_accounts": len(data.get("accounts", [])),
            "total_goals": len(data.get("goals", [])),
            "user_id": 123
        }
        
        
        transactions = data.get("transactions", [])
        if transactions:
            total_spent = sum(t["amount"] for t in transactions if t["amount"] < 0)
            total_received = sum(t["amount"] for t in transactions if t["amount"] > 0)
            stats.update({
                "total_spent": abs(total_spent),
                "total_received": total_received,
                "categories": list(set(t["category"] for t in transactions))
            })
        
        accounts = data.get("accounts", [])
        if accounts:
            total_balance = sum(a["balance"] for a in accounts)
            stats["total_balance"] = total_balance
        
        return stats
    
    def setup_dummy_environment(self):
        print("Configurando ambiente dummy do Midas AI Service...")
        
        
        os.makedirs("./data/faiss_index", exist_ok=True)
        os.makedirs("./data/tfidf_index", exist_ok=True)
        
        
        self.create_faiss_index()
        self.create_tfidf_index()
        self.generate_summary_stats()
        
        stats = self.generate_summary_stats()
        
        print("\nEstatísticas dos Dados Dummy:")
        print(f"User ID: {stats['user_id']}")
        print(f"Transações: {stats['total_transactions']}")
        print(f"Assinaturas: {stats['total_subscriptions']}")
        print(f"Contas: {stats['total_accounts']}")
        print(f"Metas: {stats['total_goals']}")
        print(f"Saldo Total: R$ {stats.get('total_balance', 0):.2f}")
        print(f"Total Gasto: R$ {stats.get('total_spent', 0):.2f}")
        print(f"Total Recebido: R$ {stats.get('total_received', 0):.2f}")
        print(f"Categorias: {', '.join(stats.get('categories', []))}")

        print("\nAmbiente dummy configurado com sucesso!")
        print("Teste com queries como:")
        print("   - 'Quanto gastei com delivery este mês?'")
        print("   - 'Quais são minhas assinaturas ativas?'")
        print("   - 'Qual é meu saldo total?'")
        print("   - 'Como estão meus cofrinhos?'")


if __name__ == "__main__":
    loader = DummyDataLoader()
    loader.setup_dummy_environment()