"""
MIDAS FDR v2 - GNN Training Script
Treina GNN com dados reais do grafo Neo4j
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

import asyncio
import torch
import torch.nn.functional as F
import numpy as np
from app.cognitive.gnn_reasoner import NeuroelasticGNN, GNNInferenceEngine
from app.cognitive.neuroelastic_graph import NeuroelasticGraph
from torch_geometric.data import Data, DataLoader


async def fetch_training_data_from_neo4j(graph: NeuroelasticGraph, num_samples: int = 100):
    """
    Busca subgrafos do Neo4j para treinar GNN, garantindo alinhamento de nó-feature-target.
    """
    training_data = []
    
    # 1. Busca dos nós iniciais (sem alteração)
    async with graph.driver.session() as session:
        result = await session.run("""
            MATCH (t:Transaction)
            WHERE t.embedding IS NOT NULL
            RETURN t.id AS id
            LIMIT $limit
        """, limit=num_samples)
        nodes = [dict(record) async for record in result]
    
    print(f"Fetched {len(nodes)} nodes from Neo4j")
    
    # Para cada nó, criar subgrafo k-hop
    for node in nodes[:20]:  # Limitar para PoC
        try:
            # 2. Busca ALINHADA no Neo4j (INCLUINDO o nó START)
            async with graph.driver.session() as session:
                # Modificação da query:
                # - Inclui o nó inicial 'start' no conjunto de nós para features e targets.
                # - Garante que 'embedding' e 'access_count' estão alinhados por nó.
                result = await session.run("""
                    MATCH (start:Transaction {id: $start_id})
                    WHERE start.embedding IS NOT NULL
                    OPTIONAL MATCH (start)-[*1..2]-(neighbor)
                    WHERE neighbor.embedding IS NOT NULL

                    // Coletar o nó inicial e seus vizinhos
                    WITH collect(start) + collect(neighbor) AS all_nodes
                    
                    // Desempacotar e usar DISTINCT para evitar duplicatas, LIMITANDO a 10 nós
                    UNWIND all_nodes AS n
                    WITH DISTINCT n
                    LIMIT 10 
                    
                    // Retornar os dados alinhados
                    RETURN n.id AS id, n.embedding AS embedding, n.access_count AS access_count
                """, start_id=node["id"])
                
                data_records = [dict(record) async for record in result]
            
            num_nodes = len(data_records)

            if num_nodes < 2:
                continue

            # 3. Construir Features e Targets ALINHADOS
            
            # features e targets são extraídos de forma alinhada
            embeddings = [r['embedding'] for r in data_records]
            access_counts = [r['access_count'] for r in data_records]
            node_ids = [r['id'] for r in data_records] # Útil para debug e mapeamento

            node_features = np.array(embeddings)
            x = torch.tensor(node_features, dtype=torch.float32)
            
            # Edge index (AGORA CORRETO: usa os IDs e o mapeamento local)
            # É necessário remapear os IDs do Neo4j para índices locais (0, 1, 2, ...)
            node_id_to_local_idx = {node_id: i for i, node_id in enumerate(node_ids)}
            
            # BUSCA DE ARESTAS (Necessário nova busca para mapear arestas corretamente)
            # Para simplificar na PoC, vamos usar a estratégia do seu grafo completo
            # (se a GNN for simples), mas o correto seria buscar as arestas internas
            # entre os nós coletados.
            
            # --- Estratégia de Grafo Completo (Simplificada) ---
            edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edges.append([i, j])
            
            if not edges:
                continue
            # ----------------------------------------------------

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # 4. Target (com o número de nós correto)
            access_counts_norm = np.array(access_counts)
            min_val = access_counts_norm.min()
            max_val = access_counts_norm.max()
            # Garante que a divisão por zero não ocorra se todos os counts forem iguais
            access_counts_norm = (access_counts_norm - min_val) / (max_val - min_val + 1e-6)
            
            # `y` terá tamanho [num_nodes, 1], que é o esperado pelo PyG para predição por nó
            y = torch.tensor(access_counts_norm, dtype=torch.float32).unsqueeze(1)
            
            print(f"✓ Subgraph {node['id']} has {num_nodes} nodes. x.size(0)={x.size(0)}, y.size(0)={y.size(0)}")
            
            training_data.append(Data(x=x, edge_index=edge_index, y=y))
        
        except Exception as e:
            print(f"❌ Error processing node {node['id']}: {e}")
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
            print(f"Batch size (nodes): {batch.x.size(0)}")
            print(f"Batch size (labels): {batch.y.size(0)}")
            print(f"Batch node features (x): {batch.x[:5]}")  # Exibir as primeiras 5 entradas
            print(f"Batch labels (y): {batch.y[:5]}")  # Exibir as primeiras 5 labels

            
            # Forward
            node_relevance, _, _ = model(batch.x, batch.edge_index, batch=batch.batch)
            # Verificando as dimensões antes de calcular a perda
            print(f"node_relevance size: {node_relevance.size()}")
            print(f"batch.y size: {batch.y.size()}")

            node_relevance = node_relevance.view(-1, 1)
            # Se os tamanhos não forem compatíveis, você pode tentar ajustar
            assert node_relevance.size(0) == batch.y.size(0), "Mismatch in batch sizes between node_relevance and batch.y"

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
