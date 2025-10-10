# MIDAS FDR - Financial Deep Research Engine

## Resumo Executivo

**Projeto:** Midas AI - Financial Deep Research  
**Contexto:** Projeto acadêmico FIAP (Análise e Desenvolvimento de Sistemas)  
**Data:** Outubro 2025

---

## Visão Geral

O **FDR (Financial Deep Research)** é uma evolução do sistema RAG tradicional do Midas, implementando uma arquitetura multi-retriever para responder queries financeiras que exigem análise de padrões e raciocínio sobre múltiplas transações.

---

## 🏗️ Arquitetura de Alto Nível

```
                    ┌─────────────────────────────┐
                    │   LangGraph Orchestrator    │
                    │   (Intent + Complexity)     │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │  Vectorial   │  │  Graph RAG   │  │    GFQR      │
      │     RAG      │  │   (Neo4j)    │  │  (GNN-based) │
      │ FAISS/pgvec  │  │              │  │              │
      └──────────────┘  └──────────────┘  └──────────────┘
              │                ▼                │
              └────────────────┼────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Weighted Fusion +   │
                    │  Multi-hop Reasoning │
                    └──────────────────────┘
```

---

## 🧠 Os 3 Retrievers Especializados

### 1. **Vectorial RAG** (Busca Semântica Tradicional)

**Quando usar:** Queries simples de lookup
- *"Quanto gastei com Uber mês passado?"*
- *"Qual meu saldo atual?"*

**Stack:**
- FAISS (in-memory, low latency)
- PostgreSQL + pgvector (persistent storage)
- OpenAI embeddings (text-embedding-3-large)

**Performance:** ~200-500ms

---

### 2. **Graph RAG** (Análise de Padrões e Relações)

**Quando usar:** Queries de tendência, padrões temporais, correlações
- *"Meus gastos com delivery estão aumentando?"*
- *"Qual a relação entre transporte e horários?"*

**Stack:**
- Neo4j (knowledge graph)
- Cypher queries dinâmicas
- PostgreSQL NER store (entidades extraídas)

**Capacidades:**
- Detecção de padrões recorrentes
- Análise de co-ocorrências (ex: "Uber depois de bar")
- Temporal reasoning ("todo dia 15 é cobrado X")
- Graph traversal multi-hop

**Performance:** ~300-800ms

**Schema Neo4j:**
```cypher
(:Transaction)-[:FROM_MERCHANT]->(:Merchant)
(:Transaction)-[:BELONGS_TO]->(:Category)
(:Transaction)-[:SIMILAR_TO {score}]->(:Transaction)
(:Subscription)-[:IMPACTS]->(:Goal)
(:Pattern {type, frequency, confidence})
```

---

### 3. **GFQR - Graph-based Financial Query Reasoning** (O Diferencial!)

**O que é:**
Sistema de raciocínio ML-based que combina:
- **GNN (Graph Neural Networks)** para entender estrutura do grafo financeiro
- **Multi-hop reasoning** para inferências complexas
- **What-if scenarios** com cálculos financeiros

**Quando usar:** Queries que exigem raciocínio causal
- *"Se eu cancelar Netflix e Spotify, quanto sobra pra investir?"*
- *"Por que minha conta fica negativa todo dia 20?"*
- *"Qual o impacto de reduzir delivery em 50%?"*

**Arquitetura GFQR:**
```python
Query → Query Embedding
  ↓
Subgraph Extraction (Neo4j)
  ↓
GNN Encoder (PyTorch Geometric)
  ├─ Node embeddings (transactions, merchants, categories)
  ├─ Edge embeddings (relations)
  └─ Graph attention layers
  ↓
Reasoning Head (Transformer-based)
  ├─ Multi-hop inference
  ├─ Causal detection
  └─ Numeric computation
  ↓
Final Answer + Reasoning Path
```

**Exemplo de Reasoning Path:**
```
Query: "Se cancelar Spotify (R$21,90) + Netflix (R$55,90), 
        quanto sobra para meta de viagem?"

GFQR Steps:
1. [Extraction] Identifica nós: Spotify, Netflix, Goal:Viagem
2. [Computation] savings = 21.90 + 55.90 = 77.80/mês
3. [Graph Query] Goal:Viagem precisa de R$3.000 em 12 meses
4. [Reasoning] 77.80 * 12 = 933.60 → contribui 31% da meta
5. [Output] "Cancelando essas assinaturas, você economiza R$933,60/ano,
             cobrindo 31% da sua meta de viagem (faltariam R$2.066,40)"
```

**Performance:** ~500-1500ms (GNN inference)

**Stack Técnico:**
- **PyTorch Geometric** (GNN framework)
- **Ray** (distributed inference para produção)
- **Custom financial reasoning layer** (domain-specific rules)

---

## 🗄️ Arquitetura de Dados (Multi-Database)

### **Oracle** (Source of Truth)
- Transações, contas, metas, assinaturas
- Dados transacionais puros

### **PostgreSQL #1** (Embeddings Store)
```sql
CREATE EXTENSION vector;

CREATE TABLE transaction_embeddings (
  transaction_id VARCHAR(36) PRIMARY KEY,
  embedding VECTOR(1536),
  metadata JSONB
);

CREATE INDEX ON transaction_embeddings 
USING ivfflat (embedding vector_cosine_ops);
```

### **PostgreSQL #2** (NER Entities Store)
```sql
CREATE TABLE financial_entities (
  id UUID PRIMARY KEY,
  transaction_id VARCHAR(36),
  entity_type VARCHAR(50),
  entity_value TEXT,
  confidence_score FLOAT
);

CREATE TABLE entity_relations (
  entity_1_id UUID,
  entity_2_id UUID,
  relation_type VARCHAR(50),
  frequency INTEGER
);
```

### **Neo4j** (Knowledge Graph)
- Nós: Transaction, Merchant, Category, Pattern, Subscription, Goal
- Edges: FROM_MERCHANT, BELONGS_TO, SIMILAR_TO, IMPACTS, PART_OF_PATTERN

### **FAISS** (In-Memory Cache)
- Índice vetorial para low-latency retrieval
- Rebuild diário ou incremental

---

## 🔄 Pipeline de Ingestão Event-Driven

```
Oracle (nova transação)
    ↓ (RabbitMQ event)
Python FDR Worker
    ├─ Gera embedding → Postgres #1
    ├─ Extrai entidades NER → Postgres #2
    ├─ Atualiza grafo → Neo4j
    └─ Atualiza índice → FAISS
```

**Async, idempotente, resiliente**

---

## 🎭 Orchestração com LangGraph

```python
class FDROrchestrator(StateGraph):
    def __init__(self):
        self.add_node("analyze_query")
        self.add_node("route_retrievers")
        self.add_node("vectorial_rag")
        self.add_node("graph_rag")
        self.add_node("gfqr_reasoning")
        self.add_node("fuse_results")
        self.add_node("verify_quality")
        
        self.add_conditional_edges(
            "route_retrievers",
            self.should_use_retriever,
            {
                "vectorial": "vectorial_rag",
                "graph": "graph_rag",
                "gfqr": "gfqr_reasoning",
                "all": "vectorial_rag"
            }
        )
        
        self.add_conditional_edges(
            "verify_quality",
            lambda state: "refine" if state["confidence"] < 0.7 else "done"
        )
```

**Decision Logic:**
- Query simples (complexity < 0.3) → **Vectorial** apenas
- Trend analysis → **Graph RAG**
- What-if scenarios → **GFQR**
- Queries complexas → **All 3** em paralelo

---

## Comparação com RAG Tradicional

| Capability | RAG Atual | FDR | Observação |
|------------|-----------|-----|------------|
| Lookup simples | 200ms | 200ms | Sem mudança |
| Trend analysis | Limitado | Suportado | Via Graph RAG |
| Pattern detection | Não suporta | Suportado | Via Neo4j |
| What-if scenarios | Não suporta | Suportado | Via GFQR |
| Multi-hop queries | Falha | Suportado | Via GNN |
| Latência média | 250ms | ~1.5s | Trade-off aceitável |

---

## Métricas Esperadas

### Performance
- Latência P95: < 3s
- Latência média: ~1.5s
- Cache hit rate: > 60%

### Qualidade
- F1-score alvo: 0.80+ (baseline atual: 0.72)
- Cobertura de queries complexas: 70%+
- Confidence calibration (ECE): < 0.15

### Utilização Estimada
- Vectorial RAG: ~90% das queries (fast path)
- Graph RAG: ~30% das queries (padrões)
- GFQR: ~10% das queries (raciocínio complexo)

---

## GFQR - Graph-based Financial Query Reasoning

### Objetivo
RAG tradicional recupera informação similar mas não raciocina sobre ela. O GFQR adiciona uma camada de raciocínio usando Graph Neural Networks para:

1. Queries causais ("Por que X aconteceu?")
2. Cenários hipotéticos ("E se eu fizesse Y?")
3. Cálculos financeiros multi-hop
4. Inferências sobre padrões de comportamento

### **Arquitetura GNN:**
```python
class FinancialReasoningGNN(torch.nn.Module):
    def __init__(self):
        self.node_encoder = nn.Linear(feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        
        self.query_attention = MultiHeadAttention(hidden_dim, num_heads=8)
        
        self.reasoning_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, reasoning_dim)
        )
        
        self.answer_decoder = nn.Linear(reasoning_dim, vocab_size)
    
    def forward(self, graph_data, query_embedding):
        x, edge_index = graph_data.x, graph_data.edge_index
        
        x = self.node_encoder(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        x = self.query_attention(x, query_embedding)
        reasoning_repr = self.reasoning_mlp(x)
        answer = self.answer_decoder(reasoning_repr)
        
        return answer, reasoning_repr
```

### **Training Data:**
```python
examples = [
    {
        "query": "Se cancelar Spotify, quanto sobra?",
        "graph": subgraph_spotify_user_123,
        "ground_truth_answer": "R$21,90/mês, R$262,80/ano",
        "reasoning_path": ["extract_subscription", "compute_savings", "format_answer"]
    }
]
```

### **Inference Example:**
```python
query = "Qual o impacto de cortar 50% dos gastos com delivery?"

subgraph = neo4j.extract_subgraph(
    user_id=123,
    entities=["delivery"],
    hops=2
)

with torch.no_grad():
    query_emb = query_encoder.encode(query)
    answer, reasoning = gfqr_model(subgraph, query_emb)

decoded = answer_decoder.decode(answer)
```

**Output:**
```json
{
    "answer": "Você gasta R$214/mês com delivery. Reduzindo 50%, economiza R$107/mês 
               ou R$1.284/ano. Isso representaria 12% da sua renda mensal.",
    "confidence": 0.89,
    "reasoning_path": [
        "Computed current delivery spend: R$214/mês",
        "Calculated 50% reduction: R$107 savings",
        "Annualized savings: R$1.284",
        "Compared to monthly income (R$900): 12% impact"
    ]
}
```

---

## Custos Estimados

### Fase de Desenvolvimento (Academic)
- Neo4j Community Edition: Gratuito
- PostgreSQL self-hosted: Gratuito
- PyTorch + PyTorch Geometric: Open source
- LangGraph: Open source
- FAISS: Open source
- GPU training (Colab Pro): ~R$50/mês

**Total desenvolvimento:** R$50-100/mês

### Produção (estimativa futura)
- Neo4j managed: ~$300/mês (starter)
- PostgreSQL managed: ~$200/mês
- GPU inference: ~$150/mês (shared)
- Infraestrutura: ~$100/mês

**Total produção:** ~$750/mês (viável para MVP)

---

## Roadmap de Implementação

### Sprint 2 (próximas 2 semanas)
- Setup Neo4j Community Edition
- Implementar pipeline de ingestão Oracle → Neo4j
- Criar GraphRAGRetriever básico
- LangGraph orchestrator (versão inicial)

### Sprint 3 (semanas 3-4)
- PostgreSQL NER store
- Fine-tuning do modelo NER para entidades financeiras
- Treinamento inicial do GFQR GNN
- Integração dos 3 retrievers

### Sprint 4 (semanas 5-6)
- Otimização de pesos do fusion
- Testes end-to-end
- Calibração de confiança
- Documentação final

---

## Diferencial do Projeto

Este projeto se diferencia de RAG tradicionais por:

- **Multi-retriever**: Três estratégias especializadas ao invés de uma genérica
- **Graph reasoning**: Uso de Neo4j para análise de padrões
- **ML-based reasoning**: GNN customizada para domínio financeiro
- **Production-grade**: Arquitetura pensada para escala (event-driven, cache, observability)

Comparado com projetos acadêmicos típicos (RAG simples com OpenAI API), demonstra:
- Conhecimento de arquitetura de sistemas
- Implementação de modelos customizados
- Design de sistemas distribuídos

---

## Resumo Técnico

O Midas FDR é uma evolução do sistema RAG atual que adiciona capacidades de raciocínio através de uma arquitetura multi-retriever orquestrada. Utilizando Neo4j para análise de grafos e uma GNN customizada para reasoning, o sistema pode responder queries financeiras complexas que exigem análise de padrões e inferências multi-hop.

**Stack principal:** FastAPI + LangGraph + Neo4j + PyTorch Geometric + PostgreSQL (pgvector)  
**Complexidade:** Alta (sistema distribuído, ML customizado, orquestração de estado)  
**Escopo:** Projeto acadêmico com arquitetura production-ready

---

*Última atualização: 10/10/2025*
