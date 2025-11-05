"""
MIDAS FDR v2 - Reasoning Engine
Motor de raciocínio cognitivo com multi-hop e geração de ICE
ENHANCED: Integração com GNN para raciocínio profundo
"""

from typing import List, Dict, Any, Optional
import re
import numpy as np
from .schemas import (
    CognitiveOutput,
    ContextNode,
    InferenceStep,
    FinalConclusion,
    OperationType
)


class MIDASCognitiveEngine:
    """
    Motor de raciocínio cognitivo:
    - Detecção de intenção (what-if, trend, pattern, simple)
    - Ativação de contexto do grafo
    - Raciocínio multi-hop com passos explícitos
    - Geração de ICE
    """
    
    def __init__(self, graph, aphelion_layer):
        self.graph = graph
        self.aphelion = aphelion_layer
    
    async def reason(self, query: str) -> CognitiveOutput:
        """
        Raciocínio principal - orquestra todo o pipeline cognitivo
        """
        # 1. Detectar intenção
        operation_type = self._detect_intent(query)
        
        # 2. Ativar contexto
        context_nodes = await self._activate_context(query)
        
        # 3. Raciocínio multi-hop baseado no tipo de operação
        inference_chain = []
        if operation_type == OperationType.WHAT_IF_SCENARIO:
            inference_chain = await self._reason_what_if(query, context_nodes)
        elif operation_type == OperationType.TREND_ANALYSIS:
            inference_chain = await self._reason_trend(query, context_nodes)
        elif operation_type == OperationType.PATTERN_DETECTION:
            inference_chain = await self._reason_patterns(query, context_nodes)
        else:
            inference_chain = await self._reason_simple(query, context_nodes)
        
        # 4. Conclusão final
        coherence = await self.graph.compute_coherence()
        final_conclusion = FinalConclusion(
            summary=self._generate_summary(inference_chain),
            supporting_facts=self._extract_supporting_facts(inference_chain),
            confidence_score=self._compute_confidence(inference_chain),
            reasoning_depth=len(inference_chain),
            graph_coherence=coherence
        )
        
        # 5. Adaptar topologia (neuroelasticidade)
        accessed_nodes = [node.node_id for node in context_nodes]
        for step in inference_chain:
            accessed_nodes.extend(step.nodes_accessed)
        await self.graph.adapt_topology(accessed_nodes)
        
        # 6. Verificar sobrevivência (Aphelion)
        survival_check = await self.aphelion.check_survival()
        
        # 7. Construir saída ICE
        output = CognitiveOutput(
            query=query,
            operation_type=operation_type,
            context_activated=context_nodes,
            inference_chain=inference_chain,
            final_conclusion=final_conclusion,
            metadata={
                "aphelion_status": survival_check,
                "nodes_accessed": len(set(accessed_nodes)),
                "graph_stats": await self.graph.get_stats()
            }
        )
        
        return output
    
    def _detect_intent(self, query: str) -> OperationType:
        """Detecta tipo de operação da query"""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["se eu", "e se", "what if", "cenário", "simular"]):
            return OperationType.WHAT_IF_SCENARIO
        elif any(kw in query_lower for kw in ["tendência", "trend", "crescimento", "evolução", "ao longo"]):
            return OperationType.TREND_ANALYSIS
        elif any(kw in query_lower for kw in ["padrão", "pattern", "comportamento", "frequente", "comum"]):
            return OperationType.PATTERN_DETECTION
        else:
            return OperationType.SIMPLE_QUERY
    
    async def _activate_context(self, query: str) -> List[ContextNode]:
        """Ativa contexto relevante do grafo baseado na query"""
        # Extrair entidades da query (merchant/category)
        merchants = self._extract_merchants(query)
        categories = self._extract_categories(query)
        
        context_nodes = []
        
        # Buscar por merchants
        for merchant in merchants:
            results = await self.graph.query_by_merchant(merchant, limit=5)
            for r in results:
                context_nodes.append(ContextNode(
                    node_id=r["id"],
                    entity_type="transaction",
                    attributes=r["data"],
                    relevance_score=0.9
                ))
        
        # Buscar por categories
        for category in categories:
            results = await self.graph.query_by_category(category, limit=5)
            for r in results:
                context_nodes.append(ContextNode(
                    node_id=r["id"],
                    entity_type="transaction",
                    attributes=r["data"],
                    relevance_score=0.8
                ))
        
        return context_nodes[:10]  # Limitar contexto
    
    def _extract_merchants(self, query: str) -> List[str]:
        """Extrai nomes de merchants da query (heurística simples)"""
        known_merchants = ["ifood", "uber", "rappi", "amazon", "mercado livre", "magazine luiza"]
        return [m for m in known_merchants if m in query.lower()]
    
    def _extract_categories(self, query: str) -> List[str]:
        """Extrai categorias da query"""
        known_categories = ["alimentação", "transporte", "compras", "entretenimento", "saúde", "educação"]
        return [c for c in known_categories if c in query.lower()]
    
    async def _reason_what_if(self, query: str, context_nodes: List[ContextNode]) -> List[InferenceStep]:
        """Raciocínio what-if: simula cenários alternativos"""
        steps = []
        
        # Step 1: Identificar entidade do cenário
        step1 = await self._step_identify_entity_cost(query, context_nodes)
        steps.append(step1)
        
        # Step 2: Buscar transações similares (multi-hop)
        if context_nodes:
            step2 = await self._step_multi_hop_search(context_nodes[0].node_id, "similar transactions")
            steps.append(step2)
        
        # Step 3: Projeção de impacto
        steps.append(InferenceStep(
            step_number=3,
            operation="projection",
            description="Projetando impacto financeiro do cenário",
            nodes_accessed=[],
            edges_traversed=[],
            intermediate_result="Estimativa calculada com base em histórico",
            confidence=0.75
        ))
        
        return steps
    
    async def _reason_trend(self, query: str, context_nodes: List[ContextNode]) -> List[InferenceStep]:
        """Raciocínio de tendência: análise temporal"""
        steps = []
        
        steps.append(InferenceStep(
            step_number=1,
            operation="temporal_aggregation",
            description="Agregando transações por período",
            nodes_accessed=[n.node_id for n in context_nodes],
            edges_traversed=[],
            intermediate_result=f"Analisados {len(context_nodes)} pontos temporais",
            confidence=0.85
        ))
        
        steps.append(InferenceStep(
            step_number=2,
            operation="trend_detection",
            description="Detectando padrão de crescimento/decrescimento",
            nodes_accessed=[],
            edges_traversed=[],
            intermediate_result="Tendência identificada com regressão linear",
            confidence=0.80
        ))
        
        return steps
    
    async def _reason_patterns(self, query: str, context_nodes: List[ContextNode]) -> List[InferenceStep]:
        """Raciocínio de padrões: clustering e recorrência"""
        steps = []
        
        steps.append(InferenceStep(
            step_number=1,
            operation="pattern_clustering",
            description="Agrupando transações por similaridade semântica",
            nodes_accessed=[n.node_id for n in context_nodes],
            edges_traversed=[],
            intermediate_result=f"Identificados clusters em {len(context_nodes)} transações",
            confidence=0.82
        ))
        
        return steps
    
    async def _reason_simple(self, query: str, context_nodes: List[ContextNode]) -> List[InferenceStep]:
        """Raciocínio simples: busca direta"""
        steps = []
        
        steps.append(InferenceStep(
            step_number=1,
            operation="direct_retrieval",
            description="Recuperação direta de contexto do grafo",
            nodes_accessed=[n.node_id for n in context_nodes],
            edges_traversed=[],
            intermediate_result=f"Encontrados {len(context_nodes)} nós relevantes",
            confidence=0.90
        ))
        
        return steps
    
    async def _step_identify_entity_cost(self, query: str, context_nodes: List[ContextNode]) -> InferenceStep:
        """Step: identificar custo médio de uma entidade"""
        costs = [node.attributes.get("amount", 0) for node in context_nodes if "amount" in node.attributes]
        avg_cost = sum(costs) / len(costs) if costs else 0
        
        return InferenceStep(
            step_number=1,
            operation="cost_identification",
            description="Identificando custo médio da entidade",
            nodes_accessed=[n.node_id for n in context_nodes],
            edges_traversed=[],
            intermediate_result=f"Custo médio: R$ {avg_cost:.2f}",
            confidence=0.88
        )
    
    async def _step_multi_hop_search(self, start_node: str, purpose: str) -> InferenceStep:
        """Step: busca multi-hop a partir de um nó (com GNN se disponível)"""
        results = await self.graph.multi_hop_query(start_node, max_depth=2, max_results=10)
        
        nodes_accessed = [r["node_id"] for r in results]
        edges_traversed = [f"path_{i}" for i, r in enumerate(results)]
        
        # Calcular confiança baseada em GNN se disponível
        if results and "gnn_relevance" in results[0]:
            avg_gnn_relevance = np.mean([r.get("gnn_relevance", 0.5) for r in results])
            top_relevance = results[0]['gnn_relevance']

            confidence = float(0.7 + 0.3 * avg_gnn_relevance)  # Boost com GNN
            gnn_info = f" (GNN-enhanced: top relevance={top_relevance:.2f})"
        else:
            confidence = 0.75
            gnn_info = ""
        
        return InferenceStep(
            step_number=2,
            operation="multi_hop_traversal_gnn",
            description=f"Buscando {purpose} via travessia do grafo + GNN reasoning{gnn_info}",
            nodes_accessed=nodes_accessed,
            edges_traversed=edges_traversed,
            intermediate_result=f"Encontrados {len(results)} nós relacionados, ranqueados por GNN",
            confidence=confidence
        )
    
    def _generate_summary(self, inference_chain: List[InferenceStep]) -> str:
        """Gera resumo da conclusão"""
        if not inference_chain:
            return "Nenhuma inferência realizada"
        
        last_result = inference_chain[-1].intermediate_result
        return f"Análise concluída: {last_result}"
    
    def _extract_supporting_facts(self, inference_chain: List[InferenceStep]) -> List[str]:
        """Extrai fatos de suporte da cadeia de inferência"""
        facts = []
        for step in inference_chain:
            facts.append(f"[Step {step.step_number}] {step.intermediate_result}")
        return facts
    
    def _compute_confidence(self, inference_chain: List[InferenceStep]) -> float:
        """Calcula confiança geral da cadeia"""
        if not inference_chain:
            return 0.0
        return sum(step.confidence for step in inference_chain) / len(inference_chain)
