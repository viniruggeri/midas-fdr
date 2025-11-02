# **MIDAS FDR 2 — Financial Deep Reasoning**

> *Beyond retrieval. Persistent inferential reasoning.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.13-green.svg)](https://neo4j.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**📄 [Whitepaper](docs/whitepaper-fdr.md) | 🚀 [Quick Start](docs/QUICK_START.md) | 📊 [MVP PoC](docs/MVP_PROOF_OF_CONCEPT.md) | 📝 [Changelog](docs/CHANGELOG.md) | 📁 [Project Structure](PROJECT_STRUCTURE.md)**

---

## 🧠 **Overview**

**MIDAS FDR 2 (Financial Deep Reasoning)** is a neuroelastic cognitive framework for persistent inferential reasoning based on dynamic topological graphs.

Unlike traditional RAG (Retrieval-Augmented Generation) systems that rely on static embeddings, **FDR 2 maintains neuroelastic contextual persistence** — allowing reasoning to evolve without loss of semantic coherence.

The system introduces a **Deep Reasoning Layer (DRL)** capable of:
- 🔗 **Persistent context** through graph topology
- 🧬 **Neuroelastic adaptation** inspired by biological neuroplasticity
- 🌊 **Multi-hop reasoning** with GNN-enhanced inference
- 🔄 **Self-healing** via Aphelion Layer (extinction/rebirth cycles)
- 💭 **What-if scenarios** with financial simulation

**Key Innovation:** The system does not summarize — it **thinks inferentially**.

### Academic Context

**Project:** FIAP (Análise e Desenvolvimento de Sistemas) — Sprint 2  
**Author:** Vinícius Ruggeri  
**Date:** November 2025  
**Whitepaper:** [whitepaper-fdr.md](whitepaper-fdr.md)

---

## 🚀 **Quick Start**

### Prerequisites
- Docker Desktop (for Neo4j)
- Python 3.10+
- 8GB RAM minimum

### Automated Setup (Windows)
```powershell
.\scripts\setup.ps1
```

### Manual Setup
```bash
# 1. Start Neo4j
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 4. Start server
python -m uvicorn app.main:app --reload --port 8000

# 5. Populate graph
curl -X POST http://localhost:8000/graph/populate

# 6. Train GNN
python scripts/train_gnn.py

# 7. Test system
python scripts/demo_mvp.py
```

**Full guide:** [docs/QUICK_START.md](docs/QUICK_START.md)

---

## 🔬 **Core Concepts**

### 1. Neuroelastic Reasoning
Inspired by biological neuroplasticity, FDR 2 maintains **adaptive connections** between contexts:
- Connections **expand**, **retract**, or **reconfigure** based on semantic flow
- New contexts don't overwrite — they **realign** through elastic re-weighting
- Preserves **persistent meaning** while remaining **adaptively plastic**

### 2. Persistent Inferential Graphs
Each node = tokenized context | Each edge = active inference

```
Traditional RAG: Query → Embed → Retrieve → Generate → Forget
FDR 2:          Query → Graph → Reason → Generate → Persist
```

### 3. Aphelion Layer (Semantic Survival)
When global coherence collapses, the system undergoes **controlled extinction and rebirth**:
- Extract core concepts via PageRank
- Prune low-relevance nodes
- Reconstruct graph from latent backups

### 4. GNN-Enhanced Multi-hop Reasoning
Graph Attention Networks (GAT) provide **topological awareness**:
- Node relevance scoring
- Multi-hop path discovery
- Confidence calibration based on graph structure

### 5. raRg Paradigm
**Retrieval-Augmented Reasoning Generation** (not just retrieval):
- GPT/LLM = Semantic interpreter
- FDR Graph = Contextual memory substrate
- Reasoning emerges from **topology**, not just embeddings

---

## ⚙️ **Architecture Overview**

### **Diagrama de Arquitetura**

```mermaid
graph TB
    subgraph Client Layer
        A[User Query]
    end
    
    subgraph FDR Engine
        B[LangGraph Router]
        B --> C{Intent Classification}
        C --> D[Complexity Estimator]
        D --> E[Retriever Planner]
        
        E --> F[Graph RAG]
        E --> G[Vectorial RAG]
        E --> H[GFQR GNN]
        
        F --> I[Weighted Fusion]
        G --> I
        H --> I
        
        I --> J[Confidence Calibration]
        J --> K[Response Generator]
    end
    
    subgraph Data Layer
        L[(Oracle)]
        M[(PostgreSQL<br/>pgvector)]
        N[(PostgreSQL<br/>NER Store)]
        O[(Neo4j<br/>Knowledge Graph)]
        P[FAISS<br/>In-Memory]
    end
    
    subgraph Event Pipeline
        Q[RabbitMQ]
        R[Sync Worker]
    end
    
    L -->|CDC Events| Q
    Q --> R
    R --> M
    R --> N
    R --> O
    R --> P
    
    F --> O
    G --> M
    G --> P
    H --> O
    H --> N
    
    K --> S[Final Response]
    
    style B fill:#4A90E2
    style I fill:#7ED321
    style L fill:#F5A623
    style O fill:#BD10E0
```

### **Fluxo de Dados**

```mermaid
sequenceDiagram
    participant U as User
    participant R as Router
    participant GR as Graph RAG
    participant VR as Vectorial RAG
    participant GFQR as GFQR GNN
    participant F as Fusion Layer
    
    U->>R: "Quanto economizaria cortando delivery em 50%?"
    R->>R: Classify Intent (what-if scenario)
    R->>R: Estimate Complexity (high)
    R->>R: Plan: [graph, vectorial, gfqr]
    
    par Parallel Retrieval
        R->>GR: Extract spending pattern
        R->>VR: Find similar transactions
        R->>GFQR: Simulate 50% reduction
    end
    
    GR-->>F: Current avg: R$214/month
    VR-->>F: 28 delivery transactions
    GFQR-->>F: Projected savings: R$107/month
    
    F->>F: Weight & Merge Results
    F->>F: Calibrate Confidence (0.87)
    F->>U: "Economizaria R$107/mês (R$1.284/ano)"
```

### **Camadas Principais**

| Camada                 | Tecnologia                | Função                                                                         |
| ---------------------- | ------------------------- | ------------------------------------------------------------------------------ |
| **Storage Principal**  | **Oracle**                | Fonte de verdade das transações financeiras.                                   |
| **Embeddings Store**   | **PostgreSQL (pgvector)** | Armazena representações vetoriais para recuperação semântica eficiente.        |
| **NER Entities Store** | **PostgreSQL**            | Base dedicada para entidades financeiras extraídas via NER.                    |
| **Knowledge Graph**    | **Neo4j**                 | Representa relações entre usuários, categorias, períodos e hábitos de consumo. |

---

## 🔄 **Pipeline de Dados**

O fluxo de dados segue o modelo **Event-Driven Sync**, no qual eventos de novas transações ou atualizações no Oracle disparam rotinas de ingestão para as demais camadas.

### **Diagrama de Ingestão**

```mermaid
flowchart LR
    A[Oracle<br/>Transaction Created] -->|Event| B[RabbitMQ Queue]
    B --> C[Python Sync Worker]
    
    C --> D[NER Extractor]
    C --> E[Embedding Generator]
    C --> F[Graph Builder]
    
    D -->|Entities| G[(PostgreSQL<br/>NER Store)]
    E -->|Vectors| H[(PostgreSQL<br/>pgvector)]
    E -->|Index| I[FAISS Cache]
    F -->|Relations| J[(Neo4j)]
    
    G -.->|Feeds| K[GFQR Training]
    J -.->|Feeds| K
    
    style A fill:#F5A623
    style C fill:#4A90E2
    style K fill:#BD10E0
```

### **Etapas de Processamento**

1. **Ingestão de dados** do Oracle (transações, categorias, metadados).
2. **Extração de entidades** via modelo NER financeiro (ex: "Uber", "Spotify", "delivery").
3. **Geração de embeddings** com vetorização contextual.
4. **Atualização do grafo** no Neo4j, representando relações e padrões emergentes.
5. **Disponibilização dos dados** para os módulos de *retrieval* do FDR.

---

## 🧩 **Mecanismo FDR**

O motor FDR é composto por três *retrievers* independentes e cooperativos:

| Retriever                                        | Tipo                       | Função Principal                                        |
| ------------------------------------------------ | -------------------------- | ------------------------------------------------------- |
| **Graph RAG**                                    | Consultas Neo4j            | Detecta tendências, relações e padrões temporais.       |
| **Vectorial RAG**                                | Similaridade vetorial      | Localiza transações semanticamente próximas.            |
| **GFQR** (*Generative Financial Query Reasoner*) | GNN + Raciocínio simbólico | Executa análises hipotéticas e inferências financeiras. |

Esses módulos são coordenados por um **Router de LangGraph**, que define dinamicamente o plano de consulta (`retriever_plan`) com base na intenção e complexidade da query.

### **Arquitetura GFQR**

```mermaid
graph TD
    A[Query] --> B[Query Encoder]
    B --> C[Subgraph Extractor]
    
    C --> D[(Neo4j)]
    D --> E[Financial Subgraph]
    
    E --> F[GNN Encoder]
    F --> G[Node Embeddings]
    F --> H[Edge Embeddings]
    
    B --> I[Query Attention Layer]
    G --> I
    H --> I
    
    I --> J[Reasoning MLP]
    J --> K[Multi-hop Inference]
    J --> L[Numeric Computation]
    
    K --> M[Answer Decoder]
    L --> M
    
    M --> N[Final Answer + Reasoning Path]
    
    style F fill:#BD10E0
    style J fill:#7ED321
    style M fill:#4A90E2
```

---

## 📊 **Exemplo de Raciocínio FDR**

**Query:**

> “Por que meus gastos com delivery aumentaram 40% nos últimos 3 meses e quanto eu economizaria se pedisse só aos finais de semana?”

### **Etapas do Pipeline**

* **NER:** identifica categorias (“delivery”), períodos (“últimos 3 meses”) e ação (“economizaria”).
* **Intent Classification:** define o tipo de análise: `trend_analysis + what_if_scenario`.
* **Retriever Plan:** seleciona `["graph", "vectorial", "gfqr"]`.
* **Graph RAG:** confirma aumento de 40% nos últimos 3 meses.
* **Vectorial RAG:** identifica padrão temporal (60% dos pedidos durante a semana).
* **GFQR:** estima economia hipotética em caso de restrição a finais de semana.

---

## 🔍 **Explainability Example**

**Resposta Final:**

> “Seus gastos com delivery aumentaram de R$180 (julho) para R$252 (setembro), um crescimento de 40%.
> O padrão mostra que 60% dos pedidos ocorrem durante a semana.
> Se você restringir pedidos apenas aos finais de semana, economizará aproximadamente R$128/mês (R$1.536/ano).”

**Traço de Raciocínio (Reasoning Trace):**

* *Graph RAG → Trend Analysis*
* *Vectorial RAG → Pattern Detection*
* *GFQR → Hypothetical Reasoning & Simulation*

**Caminho de Recuperação (Retriever Path):**

```
graph → vectorial → gfqr
```

**Confiança:** 0.91
**Latência Média:** ~2.8s

---

## 🧩 **Principais Diferenciais**

| Capacidade                   | RAG Tradicional | FDR |
| ---------------------------- | --------------- | --- |
| Perguntas factuais simples   | ✅               | ✅   |
| Análise temporal de gastos   | ⚠️              | ✅   |
| Correlação entre categorias  | ❌               | ✅   |
| Cenários “e se...” (what-if) | ❌               | ✅   |
| Explicabilidade detalhada    | ⚠️              | ✅   |

O **Midas FDR** amplia o raciocínio de IA para além da recuperação de contexto — entregando **inteligência financeira explicável e preditiva**.

---

## 📘 **Resumo Conceitual**

FDR 2 represents the transition from **context compression** to **context evolution** — a shift toward synthetic intelligence where reasoning itself becomes a dynamic, adaptive structure.

> "Enquanto o RAG responde, o FDR pensa."

---

## 📊 **Project Stats**

- **Code:** ~2,687 lines of Python
- **Architecture:** 4 core cognitive modules
- **Documentation:** 5 comprehensive guides
- **Training:** 15 epochs, ~2-3 min CPU
- **Model:** GAT-based GNN, ~50k parameters, 200KB
- **Performance:** 100-1000ms per query
- **Coverage:** Simple queries, pattern detection, what-if scenarios

**Technology Stack:**
```
FastAPI + Neo4j + PyTorch Geometric + SentenceTransformers + LangChain
```

---

## 🎓 **Academic Impact**

This project demonstrates:
- ✅ Advanced system architecture (multi-layer, distributed)
- ✅ Custom ML implementation (GNN from scratch)
- ✅ Novel theoretical framework (neuroelasticity, Aphelion Layer)
- ✅ Production-ready engineering (Docker, async, error handling)
- ✅ Comprehensive documentation (whitepaper, guides, changelog)

**Sprint 2 Deliverable:** ✅ Complete MVP with demonstrable AI/ML reasoning

---

## 📬 **Contact**

**Author:** Vinícius Ruggeri  
**GitHub:** [@viniruggeri](https://github.com/viniruggeri)  
**Project:** [midas-fdr](https://github.com/viniruggeri/midas-fdr)  
**Institution:** FIAP — Análise e Desenvolvimento de Sistemas

---

## 📄 **License**

MIT License — See [LICENSE](LICENSE) for details

---

**Built with ❤️ for the future of cognitive architectures**
