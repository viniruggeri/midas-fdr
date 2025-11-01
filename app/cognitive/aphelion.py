"""
MIDAS FDR v2 - Aphelion Layer
Camada de Homeostase Semântica - Detecção e Correção de Colapso Contextual

CONCEITO:
Aphelion Layer é o mecanismo de auto-regulação do sistema, inspirado no ponto de
maior afastamento orbital. Quando o grafo acumula ruído contextual e perde coerência
semântica, esta camada executa "extinção controlada" seguida de reconstrução do
núcleo semântico.

FORMALIZAÇÃO:
    Coerência Global:
        C(G) = (1/|E|) * Σ cos(h_i, h_j)  ∀(i,j) ∈ E
    
    Condição de Extinção:
        C(G_t) < τ_survival  por  k ≥ extinction_threshold
    
    Reconstrução:
        G_{t+1} = Reconstruct(Core(G_t))
        onde Core(G_t) = top-k nós via PageRank

ANALOGIA BIOLÓGICA:
    - Monitoramento → Detecção de estresse neural
    - Limiar τ_survival → Trigger de apoptose (morte celular programada)
    - Core Concepts → Preservação de memória semântica essencial
    - Reconstrução → Neuroplasticidade (reconexão adaptativa)
"""

from typing import List, Dict, Any, Optional
import asyncio
import numpy as np


class AphelionLayer:
    """
    Aphelion Layer: Homeostase Semântica do Grafo Neuroelástico
    
    Monitora a coerência global C(G) e executa ciclos de extinção quando
    o sistema se aproxima do colapso semântico (ruído contextual, inferências
    redundantes, perda de estrutura topológica).
    
    Args:
        graph: Instância do NeuroelasticGraph
        tau_survival: Limiar crítico de coerência (default 0.3)
        extinction_threshold: Número de colapsos consecutivos antes da extinção (default 3)
    
    Attributes:
        consecutive_low_coherence: Contador de colapsos sequenciais
        extinction_history: Log de eventos de extinção com métricas
    """
    
    def __init__(
        self,
        graph,
        tau_survival: float = 0.3,
        extinction_threshold: int = 3
    ):
        self.graph = graph
        self.tau_survival = tau_survival
        self.extinction_threshold = extinction_threshold
        self.extinction_history: List[Dict[str, Any]] = []
        self.consecutive_low_coherence = 0
    
    async def check_survival(self) -> Dict[str, Any]:
        """
        Monitora coerência global e verifica condição de extinção.
        
        Implementa a regra:
            IF C(G_t) < τ_survival FOR k ≥ extinction_threshold
            THEN perform_extinction_cycle()
        
        Returns:
            Dict contendo:
                - status: "ok" | "warning" | "critical"
                - coherence: Valor atual de C(G)
                - action: Ação executada
                - consecutive_warnings: Contador de colapsos (se warning)
                - extinction_event: Dados da extinção (se critical)
        """
        coherence = await self.graph.compute_coherence()
        
        # Colapso detectado: C(G_t) < τ_survival
        if coherence < self.tau_survival:
            self.consecutive_low_coherence += 1
            
            # Limiar de extinção atingido
            if self.consecutive_low_coherence >= self.extinction_threshold:
                extinction_event = await self.perform_extinction_cycle()
                
                return {
                    "status": "critical",
                    "coherence": coherence,
                    "action": "extinction_cycle_completed",
                    "extinctions_total": len(self.extinction_history),
                    "coherence_gain": extinction_event.get("coherence_gain", 0.0),
                    "extinction_event": extinction_event
                }
            else:
                return {
                    "status": "warning",
                    "coherence": coherence,
                    "action": "monitoring_stress",
                    "consecutive_warnings": self.consecutive_low_coherence,
                    "distance_to_extinction": self.extinction_threshold - self.consecutive_low_coherence
                }
        else:
            # Sistema estável: reset contador
            self.consecutive_low_coherence = 0
            return {
                "status": "healthy",
                "coherence": coherence,
                "action": "none",
                "tau_survival": self.tau_survival
            }
    
    async def perform_extinction_cycle(self) -> Dict[str, Any]:
        """
        Executa ciclo completo de extinção e reconstrução semântica.
        
        ALGORITMO:
            1. Snapshot do estado pré-extinção
            2. Core(G_t) ← PageRank(G_t, top_k=20)
            3. Prune(G_t \ Core(G_t))  [remove nós não-core com baixo access]
            4. Reconstruct(Core(G_t))  [reconecta semântica entre cores]
            5. Validação: compute C(G_{t+1})
        
        Returns:
            Dict contendo métricas completas do evento de extinção:
                - timestamp
                - coherence_before, coherence_after, coherence_gain
                - nodes_pruned, edges_pruned
                - core_concepts (top-k PageRank)
        """
        import time
        
        # Estado PRÉ-extinção
        coherence_before = await self.graph.compute_coherence()
        stats_before = await self.graph.get_stats()
        
        # 1. Extrair núcleo semântico via PageRank
        core_concepts = await self._extract_core_concepts(top_k=20)
        
        # 2. Snapshot completo
        snapshot = {
            "timestamp": time.time(),
            "coherence_before": coherence_before,
            "stats_before": stats_before,
            "core_concepts": core_concepts,
            "num_core_nodes": len(core_concepts)
        }
        
        # 3. Poda de nós não-essenciais
        pruned_stats = await self._prune_graph(core_concepts)
        
        # 4. Reconstrução topológica (opcional: reconectar cores)
        # await self._reconstruct_semantic_links(core_concepts)
        
        # Estado PÓS-extinção
        coherence_after = await self.graph.compute_coherence()
        stats_after = await self.graph.get_stats()
        
        # Métricas do evento
        coherence_gain = coherence_after - coherence_before
        nodes_pruned = stats_before["nodes"] - stats_after["nodes"]
        edges_pruned = stats_before["edges"] - stats_after["edges"]
        
        snapshot.update({
            "coherence_after": coherence_after,
            "coherence_gain": coherence_gain,
            "stats_after": stats_after,
            "nodes_pruned": nodes_pruned,
            "edges_pruned": edges_pruned,
            "pruned_details": pruned_stats
        })
        
        # Log do evento
        self.extinction_history.append(snapshot)
        self.consecutive_low_coherence = 0
        
        print(f"[APHELION] Extinction cycle #{len(self.extinction_history)} completed:")
        print(f"  Coherence: {coherence_before:.4f} → {coherence_after:.4f} (Δ = +{coherence_gain:.4f})")
        print(f"  Pruned: {nodes_pruned} nodes, {edges_pruned} edges")
        print(f"  Core preserved: {len(core_concepts)} concepts")
        
        return snapshot
    
    async def _extract_core_concepts(self, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Extrai núcleo semântico via PageRank ponderado.
        
        Implementa Core(G_t):
            PageRank com pesos das arestas (similaridade semântica)
            → Identifica nós centrais na topologia do conhecimento
        
        Args:
            top_k: Número de conceitos core a preservar
        
        Returns:
            Lista de dicts: [{"node_id", "pagerank_score", "attributes"}]
            Ordenada por relevância (score DESC)
        """
        async with self.graph.driver.session() as session:
            try:
                # Criar projeção do grafo para GDS (Graph Data Science)
                await session.run("""
                    CALL gds.graph.project(
                        'extinction_graph',
                        'Transaction',
                        {
                            SIMILAR_TO: {orientation: 'UNDIRECTED'},
                            FROM_MERCHANT: {},
                            IN_CATEGORY: {}
                        },
                        {relationshipProperties: 'weight'}
                    )
                """)
                
                # Executar PageRank ponderado
                result = await session.run("""
                    CALL gds.pageRank.stream('extinction_graph', {
                        relationshipWeightProperty: 'weight',
                        maxIterations: 20,
                        dampingFactor: 0.85
                    })
                    YIELD nodeId, score
                    WITH gds.util.asNode(nodeId) AS node, score
                    RETURN node.id AS node_id,
                           score AS pagerank_score,
                           node {.merchant, .category, .amount, .description, .access_count} AS attributes
                    ORDER BY score DESC
                    LIMIT $top_k
                """, top_k=top_k)
                
                core_concepts = [dict(record) async for record in result]
                
                # Limpar projeção
                await session.run("CALL gds.graph.drop('extinction_graph')")
                
                return core_concepts
            
            except Exception as e:
                print(f"[APHELION] PageRank failed (GDS plugin may be missing): {e}")
                # Fallback: usar access_count como proxy
                return await self._extract_core_by_access_count(session, top_k)
    
    async def _extract_core_by_access_count(self, session, top_k: int) -> List[Dict[str, Any]]:
        """
        Fallback: extrai cores por access_count (quando GDS não disponível)
        """
        result = await session.run("""
            MATCH (n:Transaction)
            WHERE n.access_count IS NOT NULL
            RETURN n.id AS node_id,
                   n.access_count AS pagerank_score,
                   n {.merchant, .category, .amount, .description, .access_count} AS attributes
            ORDER BY n.access_count DESC
            LIMIT $top_k
        """, top_k=top_k)
        
        return [dict(record) async for record in result]
    
    async def _prune_graph(self, core_concepts: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Executa poda seletiva: Prune(G_t \ Core(G_t))
        
        Remove nós que:
            - NÃO estão em Core(G_t) (PageRank top-k)
            - E têm baixo access_count (< 2)
        
        Preserva:
            - Todos os nós core (memória semântica essencial)
            - Nós periféricos com alta utilização recente
        
        Args:
            core_concepts: Lista de conceitos core (PageRank top-k)
        
        Returns:
            Dict com estatísticas da poda: {"nodes_deleted", "edges_deleted"}
        """
        core_ids = [concept["node_id"] for concept in core_concepts]
        
        async with self.graph.driver.session() as session:
            # Contar nós/arestas antes da poda
            result = await session.run("""
                MATCH (n:Transaction)
                WHERE NOT n.id IN $core_ids
                  AND COALESCE(n.access_count, 0) < 2
                WITH count(n) AS nodes_to_delete
                MATCH (n:Transaction)-[r]-()
                WHERE NOT n.id IN $core_ids
                  AND COALESCE(n.access_count, 0) < 2
                RETURN nodes_to_delete, count(DISTINCT r) AS edges_to_delete
            """, core_ids=core_ids)
            
            stats_record = await result.single()
            nodes_to_delete = stats_record["nodes_to_delete"] if stats_record else 0
            edges_to_delete = stats_record["edges_to_delete"] if stats_record else 0
            
            # Executar poda
            await session.run("""
                MATCH (n:Transaction)
                WHERE NOT n.id IN $core_ids
                  AND COALESCE(n.access_count, 0) < 2
                DETACH DELETE n
            """, core_ids=core_ids)
            
            return {
                "nodes_deleted": nodes_to_delete,
                "edges_deleted": edges_to_delete
            }
    
    async def get_extinction_summary(self) -> Dict[str, Any]:
        """
        Retorna análise estatística completa dos eventos de extinção.
        
        Returns:
            Dict contendo:
                - total_extinctions: Número de ciclos executados
                - avg_coherence_gain: Ganho médio de C(G) por extinção
                - total_nodes_pruned: Soma de nós removidos
                - last_extinction: Dados do evento mais recente
                - system_health: Status atual do sistema
        """
        if not self.extinction_history:
            return {
                "total_extinctions": 0,
                "message": "No extinction events recorded",
                "system_health": "stable" if self.consecutive_low_coherence == 0 else "monitoring"
            }
        
        # Métricas agregadas
        total_coherence_gain = sum(
            e.get("coherence_gain", 0) for e in self.extinction_history
        )
        avg_coherence_gain = total_coherence_gain / len(self.extinction_history)
        
        total_nodes_pruned = sum(
            e.get("nodes_pruned", 0) for e in self.extinction_history
        )
        
        total_edges_pruned = sum(
            e.get("edges_pruned", 0) for e in self.extinction_history
        )
        
        return {
            "total_extinctions": len(self.extinction_history),
            "avg_coherence_gain": round(avg_coherence_gain, 4),
            "total_nodes_pruned": total_nodes_pruned,
            "total_edges_pruned": total_edges_pruned,
            "last_extinction": {
                "timestamp": self.extinction_history[-1].get("timestamp"),
                "coherence_before": self.extinction_history[-1].get("coherence_before"),
                "coherence_after": self.extinction_history[-1].get("coherence_after"),
                "coherence_gain": self.extinction_history[-1].get("coherence_gain"),
                "nodes_pruned": self.extinction_history[-1].get("nodes_pruned"),
                "core_concepts_preserved": self.extinction_history[-1].get("num_core_nodes")
            },
            "system_health": "post_extinction_recovery" if self.consecutive_low_coherence == 0 else "monitoring"
        }
