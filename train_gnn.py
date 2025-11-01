"""
MIDAS FDR v2 - GNN Training Script
Treina GNN com dados reais do grafo Neo4j
"""

import asyncio
import torch
import torch.nn.functional as F
import numpy as np
from app.cognitive.gnn_reasoner import NeuroelasticGNN, GNNInferenceEngine
from app.cognitive.neuroelastic_graph import NeuroelasticGraph
from torch_geometric.data import Data, DataLoader


async def fetch_training_data_from_neo4j(graph: NeuroelasticGraph, num_samples: int = 100):
    """
    Busca subgrafos do Neo4j para treinar GNN
    """
    training_data = []
    
    async with graph.driver.session() as session:
        # Buscar nós com embeddings
        result = await session.run("""
            MATCH (t:Transaction)
            WHERE t.embedding IS NOT NULL
            RETURN t.id AS id, t.embedding AS embedding, t.access_count AS access_count
            LIMIT $limit
        """, limit=num_samples)
        
        nodes = [dict(record) async for record in result]
    
    print(f"Fetched {len(nodes)} nodes from Neo4j")
    
    # Para cada nó, criar subgrafo k-hop
    for node in nodes[:20]:  # Limitar para PoC
        try:
            # Buscar vizinhança
            async with graph.driver.session() as session:
                result = await session.run("""
                    MATCH path = (start:Transaction {id: $start_id})-[*1..2]-(neighbor)
                    WHERE neighbor.embedding IS NOT NULL
                    RETURN collect(DISTINCT neighbor.id) AS neighbors,
                           collect(DISTINCT neighbor.embedding) AS embeddings,
                           collect(DISTINCT neighbor.access_count) AS access_counts
                    LIMIT 20
                """, start_id=node["id"])
                
                record = await result.single()
                if not record:
                    continue
                
                neighbors = record["neighbors"]
                embeddings = record["embeddings"]
                access_counts = record["access_counts"]
            
            if len(neighbors) < 2:
                continue
            
            # Construir features
            node_features = np.array(embeddings[:10])  # Max 10 nós
            
            # Edge index (grafo completo para simplificar)
            num_nodes = len(node_features)
            edges = []
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    edges.append([i, j])
                    edges.append([j, i])
            
            if not edges:
                continue
            
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            x = torch.tensor(node_features, dtype=torch.float32)
            
            # Target: nós com alto access_count são relevantes
            access_counts_norm = np.array(access_counts[:10])
            access_counts_norm = (access_counts_norm - access_counts_norm.min()) / (access_counts_norm.max() - access_counts_norm.min() + 1e-6)
            y = torch.tensor(access_counts_norm, dtype=torch.float32).unsqueeze(1)
            
            training_data.append(Data(x=x, edge_index=edge_index, y=y))
        
        except Exception as e:
            print(f"Error processing node {node['id']}: {e}")
            continue
    
    return training_data


async def train_gnn_on_real_data():
    """
    Treina GNN com dados reais do Neo4j
    """
    print("=" * 80)
    print("MIDAS FDR v2 - GNN Training on Real Data")
    print("=" * 80)
    
    # Conectar ao grafo
    graph = NeuroelasticGraph()
    
    # Buscar dados de treino
    print("\n1. Fetching training data from Neo4j...")
    training_data = await fetch_training_data_from_neo4j(graph)
    
    if not training_data:
        print("❌ No training data found. Populate graph first with: POST /graph/populate")
        await graph.close()
        return
    
    print(f"✓ Collected {len(training_data)} training samples")
    
    # Criar DataLoader
    loader = DataLoader(training_data, batch_size=4, shuffle=True)
    
    # Modelo
    model = NeuroelasticGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    print("\n2. Training GNN...")
    model.train()
    
    for epoch in range(15):
        total_loss = 0
        
        for batch in loader:
            optimizer.zero_grad()
            
            # Forward
            node_relevance, _, _ = model(batch.x, batch.edge_index, batch=batch.batch)
            
            # Loss
            loss = F.mse_loss(node_relevance, batch.y)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/15 - Loss: {avg_loss:.4f}")
    
    # Salvar modelo
    torch.save(model.state_dict(), "gnn_neuroelastic_pretrained.pt")
    print("\n✓ Model saved to: gnn_neuroelastic_pretrained.pt")
    
    # Testar modelo
    print("\n3. Testing trained model...")
    model.eval()
    engine = GNNInferenceEngine("gnn_neuroelastic_pretrained.pt")
    
    test_sample = training_data[0]
    result = engine.infer(test_sample)
    
    print(f"   Graph confidence: {result['graph_confidence']:.3f}")
    print(f"   Top-3 nodes: {result['top_k_nodes'][:3]}")
    
    await graph.close()
    
    print("\n" + "=" * 80)
    print("✓ GNN Training Completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(train_gnn_on_real_data())
