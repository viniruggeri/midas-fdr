"""
MIDAS FDR v2 - ICE Schemas
Interface Cognitiva Estruturada - Data Models
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json


class OperationType(str, Enum):
    """Tipos de operação cognitiva"""
    SIMPLE_QUERY = "simple_query"
    WHAT_IF_SCENARIO = "what_if_scenario"
    TREND_ANALYSIS = "trend_analysis"
    PATTERN_DETECTION = "pattern_detection"


@dataclass
class ContextNode:
    """Nó de contexto extraído do grafo"""
    node_id: str
    entity_type: str  # "merchant", "category", "transaction"
    attributes: Dict[str, Any]
    relevance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "entity_type": self.entity_type,
            "attributes": self.attributes,
            "relevance_score": self.relevance_score
        }


@dataclass
class InferenceStep:
    """Passo de raciocínio multi-hop"""
    step_number: int
    operation: str
    description: str
    nodes_accessed: List[str]
    edges_traversed: List[str]
    intermediate_result: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "operation": self.operation,
            "description": self.description,
            "nodes_accessed": self.nodes_accessed,
            "edges_traversed": self.edges_traversed,
            "intermediate_result": self.intermediate_result,
            "confidence": self.confidence
        }


@dataclass
class FinalConclusion:
    """Conclusão final do raciocínio"""
    summary: str
    supporting_facts: List[str]
    confidence_score: float
    reasoning_depth: int
    graph_coherence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "supporting_facts": self.supporting_facts,
            "confidence_score": self.confidence_score,
            "reasoning_depth": self.reasoning_depth,
            "graph_coherence": self.graph_coherence
        }


@dataclass
class CognitiveOutput:
    """Saída estruturada da Interface Cognitiva Estruturada (ICE)"""
    query: str
    operation_type: OperationType
    context_activated: List[ContextNode] = field(default_factory=list)
    inference_chain: List[InferenceStep] = field(default_factory=list)
    final_conclusion: Optional[FinalConclusion] = None
    humanized_response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "operation_type": self.operation_type.value,
            "context_activated": [node.to_dict() for node in self.context_activated],
            "inference_chain": [step.to_dict() for step in self.inference_chain],
            "final_conclusion": self.final_conclusion.to_dict() if self.final_conclusion else None,
            "humanized_response": self.humanized_response,
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
