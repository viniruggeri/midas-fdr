"""
MIDAS FDR v2 - GNN Reasoner
Graph Neural Network para raciocínio multi-hop REAL
Proof of Concept com PyTorch Geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Dict, Any, Tuple


class NeuroelasticGNN(nn.Module):
    """
    GNN para raciocínio neuroelástico:
    - 2 camadas GAT (Graph Attention) para capturar relevância contextual
    - Aggregation global para gerar embedding do subgrafo
    - Predição de confiança e relevância de nós
    """
    
    def __init__(
        self,
        input_dim: int = 384,  # Dimensão do SentenceTransformer
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_heads: int = 4
    ):
        super().__init__()
        
        # Layer 1: GAT com multi-head attention
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        
        # Layer 2: GAT para refinar
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)
        
        # MLP para predição de relevância de nós
        self.node_relevance = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # MLP para predição de confiança do subgrafo
        self.graph_confidence = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges] (opcional)
            batch: Batch assignment [num_nodes] (para batching)
        
        Returns:
            node_relevance: [num_nodes, 1]
            graph_confidence: [batch_size, 1]
            node_embeddings: [num_nodes, output_dim]
        """
        # GAT Layer 1
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        
        # GAT Layer 2
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        
        # Predição de relevância por nó
        node_relevance = self.node_relevance(h)
        
        # Aggregation global para confiança do grafo
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        graph_embedding = global_mean_pool(h, batch)
        graph_confidence = self.graph_confidence(graph_embedding)
        
        return node_relevance, graph_confidence, h


class GNNInferenceEngine:
    """
    Motor de inferência usando GNN treinada
    """
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NeuroelasticGNN().to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
    
    def create_subgraph_from_context(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_weights: np.ndarray = None
    ) -> Data:
        """
        Converte contexto do grafo para PyTorch Geometric Data
        
        Args:
            node_features: [num_nodes, feature_dim]
            edge_index: [2, num_edges] (source, target)
            edge_weights: [num_edges]
        """
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        if edge_weights is not None:
            edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            edge_weight = None
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
    
    @torch.no_grad()
    def infer(self, subgraph: Data) -> Dict[str, Any]:
        """
        Executa inferência no subgrafo
        
        Returns:
            {
                "node_relevance": [num_nodes],
                "graph_confidence": float,
                "top_k_nodes": [(node_idx, relevance_score)],
                "node_embeddings": [num_nodes, output_dim]
            }
        """
        subgraph = subgraph.to(self.device)
        
        node_relevance, graph_confidence, node_embeddings = self.model(
            subgraph.x,
            subgraph.edge_index,
            subgraph.edge_attr
        )
        
        # Converter para numpy
        node_relevance_np = node_relevance.cpu().numpy().flatten()
        graph_confidence_np = graph_confidence.cpu().item()
        node_embeddings_np = node_embeddings.cpu().numpy()
        
        # Top-k nós mais relevantes
        top_k_indices = np.argsort(node_relevance_np)[-5:][::-1]
        top_k_nodes = [(int(idx), float(node_relevance_np[idx])) for idx in top_k_indices]
        
        return {
            "node_relevance": node_relevance_np,
            "graph_confidence": graph_confidence_np,
            "top_k_nodes": top_k_nodes,
            "node_embeddings": node_embeddings_np
        }
    
    def multi_hop_reasoning(
        self,
        start_node_idx: int,
        all_node_features: np.ndarray,
        adjacency_list: Dict[int, List[int]],
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Raciocínio multi-hop usando GNN:
        1. Começa no nó inicial
        2. Expande k-hop neighborhood
        3. GNN computa relevância
        4. Seleciona top-k para próximo hop
        
        Returns:
            Lista de inferências por hop
        """
        reasoning_chain = []
        current_nodes = {start_node_idx}
        visited = set()
        
        for hop in range(max_hops):
            # Expandir vizinhança
            neighbors = set()
            for node in current_nodes:
                if node in adjacency_list:
                    neighbors.update(adjacency_list[node])
            
            subgraph_nodes = list(current_nodes | neighbors)
            visited.update(current_nodes)
            
            # Construir subgrafo
            node_features = all_node_features[subgraph_nodes]
            
            # Edge index (simplificado para PoC)
            edges = []
            for i, src in enumerate(subgraph_nodes):
                if src in adjacency_list:
                    for tgt in adjacency_list[src]:
                        if tgt in subgraph_nodes:
                            j = subgraph_nodes.index(tgt)
                            edges.append([i, j])
            
            if not edges:
                break
            
            edge_index = np.array(edges).T
            subgraph_data = self.create_subgraph_from_context(node_features, edge_index)
            
            # Inferência GNN
            inference_result = self.infer(subgraph_data)
            
            # Registrar hop
            reasoning_chain.append({
                "hop": hop + 1,
                "nodes_explored": [subgraph_nodes[idx] for idx, _ in inference_result["top_k_nodes"]],
                "relevance_scores": [score for _, score in inference_result["top_k_nodes"]],
                "graph_confidence": inference_result["graph_confidence"]
            })
            
            # Próximo hop: top-3 nós mais relevantes não visitados
            next_nodes = set()
            for idx, _ in inference_result["top_k_nodes"][:3]:
                real_idx = subgraph_nodes[idx]
                if real_idx not in visited:
                    next_nodes.add(real_idx)
            
            if not next_nodes:
                break
            
            current_nodes = next_nodes
        
        return reasoning_chain


def pretrain_gnn_simple():
    """
    Pré-treina GNN com task sintética (link prediction)
    PoC: treina em grafos dummy para demonstrar capacidade
    """
    model = NeuroelasticGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Dados sintéticos
    num_samples = 50
    num_nodes = 20
    feature_dim = 384
    
    print("Pré-treinando GNN com dados sintéticos...")
    
    for epoch in range(10):
        total_loss = 0
        
        for _ in range(num_samples):
            # Gerar grafo aleatório
            x = torch.randn(num_nodes, feature_dim)
            edge_index = torch.randint(0, num_nodes, (2, 40))
            
            # Target: nós com features similares ao nó central (idx=0) são relevantes
            target_relevance = F.cosine_similarity(x, x[0].unsqueeze(0), dim=1).unsqueeze(1)
            target_relevance = (target_relevance + 1) / 2  # Normalizar [0, 1]
            
            # Forward
            node_relevance, graph_confidence, _ = model(x, edge_index)
            
            # Loss
            loss = F.mse_loss(node_relevance, target_relevance)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/10 - Loss: {total_loss/num_samples:.4f}")
    
    # Salvar modelo
    torch.save(model.state_dict(), "gnn_neuroelastic_pretrained.pt")
    print("✓ Modelo salvo em gnn_neuroelastic_pretrained.pt")
    
    return model


if __name__ == "__main__":
    # Treinar modelo
    model = pretrain_gnn_simple()
    
    # Testar inferência
    print("\n=== Testando inferência GNN ===")
    engine = GNNInferenceEngine()
    
    # Grafo de teste
    num_nodes = 10
    node_features = np.random.randn(num_nodes, 384)
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    
    subgraph = engine.create_subgraph_from_context(node_features, edge_index)
    result = engine.infer(subgraph)
    
    print(f"Graph confidence: {result['graph_confidence']:.3f}")
    print(f"Top-3 relevant nodes: {result['top_k_nodes'][:3]}")
