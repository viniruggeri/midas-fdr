"""
MIDAS FDR v2 - NeuroelasticGraph
Grafo dinâmico com adaptação topológica e persistência em Neo4j
HYBRID: Neo4j para persistência + GNN para raciocínio
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from neo4j import AsyncGraphDatabase
from sentence_transformers import SentenceTransformer
import asyncio
from .gnn_reasoner import GNNInferenceEngine
import networkx as nx


class NeuroelasticGraph:
    """
    Grafo neuroelástico com:
    - Persistência Neo4j
    - Criação automática de arestas semânticas (cosine similarity > tau_min)
    - Adaptação topológica via neuroelasticidade
    - Query multi-hop com Cypher
    """
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        tau_min: float = 0.7,
        eta: float = 0.01,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_gnn: bool = True
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "midas123")
        self.tau_min = tau_min
        self.eta = eta
        self.use_gnn = use_gnn
        
        self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.embedder = SentenceTransformer(embedding_model)
        
        # GNN para raciocínio avançado
        if self.use_gnn:
            try:
                self.gnn_engine = GNNInferenceEngine("gnn_neuroelastic_pretrained.pt")
            except:
                self.gnn_engine = GNNInferenceEngine()  # Sem modelo pré-treinado
        else:
            self.gnn_engine = None
    
    async def close(self):
        await self.driver.close()
    
    async def add_transaction_node(
        self,
        transaction_id: str,
        merchant: str,
        category: str,
        amount: float,
        description: str,
        timestamp: str
    ) -> str:
        """
        Adiciona nó de transação com embeddings e cria arestas semânticas automaticamente
        """
        # Gerar embedding
        text = f"{merchant} {category} {description}"
        embedding = self.embedder.encode(text).tolist()
        
        async with self.driver.session() as session:
            # Criar nó
            await session.run("""
                MERGE (t:Transaction {id: $id})
                SET t.merchant = $merchant,
                    t.category = $category,
                    t.amount = $amount,
                    t.description = $description,
                    t.timestamp = $timestamp,
                    t.embedding = $embedding,
                    t.access_count = COALESCE(t.access_count, 0)
            """, id=transaction_id, merchant=merchant, category=category,
                amount=amount, description=description, timestamp=timestamp,
                embedding=embedding)
            
            # Criar arestas categóricas
            await self._create_categorical_edges(session, transaction_id, merchant, category)
            
            # Criar arestas semânticas
            await self._create_semantic_edges(session, transaction_id, embedding)
        
        return transaction_id
    
    async def _create_categorical_edges(
        self,
        session,
        transaction_id: str,
        merchant: str,
        category: str
    ):
        """Cria arestas explícitas merchant/category"""
        await session.run("""
            MATCH (t:Transaction {id: $tid})
            MERGE (m:Merchant {name: $merchant})
            MERGE (c:Category {name: $category})
            MERGE (t)-[:FROM_MERCHANT {weight: 1.0}]->(m)
            MERGE (t)-[:IN_CATEGORY {weight: 1.0}]->(c)
        """, tid=transaction_id, merchant=merchant, category=category)
    
    async def _create_semantic_edges(
        self,
        session,
        transaction_id: str,
        embedding: List[float]
    ):
        """
        Cria arestas semânticas via cosine similarity > tau_min
        """
        # Buscar candidatos (últimos 100 nós)
        result = await session.run("""
            MATCH (t:Transaction)
            WHERE t.id <> $tid AND t.embedding IS NOT NULL
            RETURN t.id AS id, t.embedding AS embedding
            ORDER BY t.timestamp DESC
            LIMIT 100
        """, tid=transaction_id)
        
        candidates = [record async for record in result]
        
        # Calcular similaridade
        embedding_np = np.array(embedding)
        for candidate in candidates:
            candidate_embedding = np.array(candidate["embedding"])
            similarity = self._cosine_similarity(embedding_np, candidate_embedding)
            
            if similarity >= self.tau_min:
                await session.run("""
                    MATCH (t1:Transaction {id: $tid1})
                    MATCH (t2:Transaction {id: $tid2})
                    MERGE (t1)-[r:SIMILAR_TO]-(t2)
                    SET r.weight = $similarity
                """, tid1=transaction_id, tid2=candidate["id"], similarity=float(similarity))
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    async def multi_hop_query(
        self,
        start_node_id: str,
        max_depth: int = 3,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Query multi-hop a partir de um nó inicial
        HYBRID: Cypher para busca inicial + GNN para ranking
        Retorna: [{"node_id", "path", "distance", "attributes", "gnn_relevance"}]
        """
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH path = (start:Transaction {id: $start_id})-[*1..""" + str(max_depth) + """]->(target)
                WITH target, path, 
                     reduce(dist = 0, r IN relationships(path) | dist + 1.0/r.weight) AS distance
                ORDER BY distance ASC
                LIMIT $max_results
                RETURN target.id AS node_id,
                       [n IN nodes(path) | n.id] AS path,
                       distance,
                       target {.merchant, .category, .amount, .description, .embedding} AS attributes
            """, start_id=start_node_id, max_results=max_results)
            
            cypher_results = [dict(record) async for record in result]
            
            # Se GNN ativada, refinar com raciocínio neural
            if self.use_gnn and self.gnn_engine and cypher_results:
                cypher_results = await self._refine_with_gnn(cypher_results)
            
            return cypher_results
    
    async def _refine_with_gnn(self, cypher_results: List[Dict]) -> List[Dict]:
        """
        Refina resultados Cypher com GNN
        """
        try:
            # Extrair features e construir subgrafo
            node_features = []
            node_ids = []
            
            for r in cypher_results:
                if "embedding" in r["attributes"] and r["attributes"]["embedding"]:
                    node_features.append(r["attributes"]["embedding"])
                    node_ids.append(r["node_id"])
            
            if len(node_features) < 2:
                return cypher_results
            
            # Criar edge_index (grafo sequencial para PoC)
            edges = [[i, i+1] for i in range(len(node_features)-1)]
            if not edges:
                return cypher_results
            
            edge_index = np.array(edges).T
            node_features_np = np.array(node_features)
            
            # Inferência GNN
            subgraph = self.gnn_engine.create_subgraph_from_context(node_features_np, edge_index)
            gnn_result = self.gnn_engine.infer(subgraph)
            
            # Adicionar relevância GNN aos resultados
            for i, result in enumerate(cypher_results[:len(node_ids)]):
                result["gnn_relevance"] = float(gnn_result["node_relevance"][i])
            
            # Re-ordenar por relevância GNN
            cypher_results.sort(key=lambda x: x.get("gnn_relevance", 0), reverse=True)
            
        except Exception as e:
            print(f"GNN refinement failed: {e}")
        
        return cypher_results
    
    async def query_by_merchant(self, merchant: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Busca transações de um merchant"""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (t:Transaction)-[:FROM_MERCHANT]->(m:Merchant {name: $merchant})
                RETURN t.id AS id, t {.merchant, .category, .amount, .description, .timestamp} AS data
                ORDER BY t.timestamp DESC
                LIMIT $limit
            """, merchant=merchant, limit=limit)
            
            return [dict(record) async for record in result]
    
    async def query_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Busca transações de uma categoria"""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (t:Transaction)-[:IN_CATEGORY]->(c:Category {name: $category})
                RETURN t.id AS id, t {.merchant, .category, .amount, .description, .timestamp} AS data
                ORDER BY t.timestamp DESC
                LIMIT $limit
            """, category=category, limit=limit)
            
            return [dict(record) async for record in result]
    
    async def adapt_topology(self, accessed_nodes: List[str]):
        """
        Implementa neuroelasticidade: dW_t/dt = η(Φ(V_t) - Λ_t)
        Incrementa pesos de arestas conectadas a nós acessados
        """
        if not accessed_nodes:
            return
        
        async with self.driver.session() as session:
            # Incrementar access_count
            await session.run("""
                UNWIND $nodes AS node_id
                MATCH (n:Transaction {id: node_id})
                SET n.access_count = COALESCE(n.access_count, 0) + 1
            """, nodes=accessed_nodes)
            
            # Atualizar pesos das arestas
            await session.run("""
                UNWIND $nodes AS node_id
                MATCH (n:Transaction {id: node_id})-[r]-()
                SET r.weight = r.weight + $eta
            """, nodes=accessed_nodes, eta=self.eta)
    
    async def compute_coherence(self) -> float:
        """
        Calcula coerência global do grafo C(G).
        
        IMPLEMENTAÇÃO HÍBRIDA:
            Método 1 (ideal): C(G) = (1/|E|) * Σ cos(h_i, h_j)  ∀(i,j) ∈ E
                → Similaridade coseno média entre embeddings conectados
            
            Método 2 (fallback): C(G) = (edge_density + clustering) / (1 + entropy)
                → Topológico (quando embeddings indisponíveis)
        
        Returns:
            float: Coerência normalizada [0, 1]
        """
        async with self.driver.session() as session:
            # Tentar método 1: Coerência semântica via embeddings
            try:
                result = await session.run("""
                    MATCH (n1:Transaction)-[r]-(n2:Transaction)
                    WHERE n1.embedding IS NOT NULL AND n2.embedding IS NOT NULL
                    WITH n1.embedding AS emb1, n2.embedding AS emb2
                    WITH reduce(s = 0.0, i IN range(0, size(emb1)-1) | s + emb1[i]*emb2[i]) AS dot_product,
                         sqrt(reduce(s = 0.0, i IN range(0, size(emb1)-1) | s + emb1[i]*emb1[i])) AS norm1,
                         sqrt(reduce(s = 0.0, i IN range(0, size(emb2)-1) | s + emb2[i]*emb2[i])) AS norm2
                    WITH dot_product / (norm1 * norm2) AS cosine_sim
                    RETURN avg(cosine_sim) AS semantic_coherence, count(*) AS edges_analyzed
                """)
                
                record = await result.single()
                if record and record["edges_analyzed"] > 0:
                    semantic_coherence = record["semantic_coherence"]
                    # Normalizar de [-1, 1] para [0, 1]
                    normalized_coherence = (semantic_coherence + 1) / 2
                    return float(normalized_coherence)
            
            except Exception as e:
                print(f"[COHERENCE] Semantic method failed, using topological fallback: {e}")
            
            # Método 2: Coerência topológica (fallback)
            # Edge density
            result = await session.run("""
                MATCH (n:Transaction)
                WITH count(n) AS node_count
                MATCH ()-[r]->()
                WITH node_count, count(r) AS edge_count
                WHERE node_count > 1
                RETURN toFloat(edge_count) / (node_count * (node_count - 1)) AS edge_density
            """)
            record = await result.single()
            edge_density = record["edge_density"] if record and record["edge_density"] else 0.0
            
            # Clustering coefficient (via GDS ou fallback)
            try:
                result = await session.run("""
                    CALL gds.localClusteringCoefficient.stats('Transaction')
                    YIELD averageClusteringCoefficient
                    RETURN averageClusteringCoefficient AS clustering
                """)
                record = await result.single()
                clustering = record["clustering"] if record else 0.0
            except:
                clustering = 0.0  # GDS não disponível
            
            # Entropy (distribuição de graus)
            result = await session.run("""
                MATCH (n:Transaction)
                WITH n, size((n)-[]-()) AS degree
                WHERE degree > 0
                WITH degree, count(*) AS freq
                WITH toFloat(freq) / sum(freq) AS prob
                WHERE prob > 0
                RETURN -sum(prob * log(prob)) AS entropy
            """)
            record = await result.single()
            entropy = record["entropy"] if record and record["entropy"] else 0.0
            
            # Fórmula topológica
            coherence = (edge_density + clustering) / (1 + entropy)
            return float(coherence)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do grafo"""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (t:Transaction)
                OPTIONAL MATCH ()-[r]->()
                RETURN count(DISTINCT t) AS nodes,
                       count(r) AS edges
            """)
            record = await result.single()
            
            coherence = await self.compute_coherence()
            
            return {
                "nodes": record["nodes"],
                "edges": record["edges"],
                "coherence": coherence
            }
