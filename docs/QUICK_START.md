# 🚀 GUIA RÁPIDO - MIDAS FDR v2

## ✅ CHECKLIST PRÉ-REQUISITOS

```powershell
# 1. Verificar Docker
docker --version
# ✅ Docker version 28.4.0 instalado

# 2. Verificar Python
python --version
# Precisa: Python 3.10+

# 3. Verificar Git
git --version
```

---

## 📋 PASSO A PASSO (15 MINUTOS)

### **1️⃣ Inicie o Docker Desktop**

```
1. Abra o Docker Desktop (ícone da baleia azul)
2. Aguarde até aparecer "Docker Desktop is running" (30-60 segundos)
3. Teste: docker ps
```

---

### **2️⃣ Configure o Ambiente**

```powershell
# Navegue até o projeto
cd C:\Users\rugge_p2gkz2r\Desktop\midas-ai\midas-ai-service

# Copie o arquivo de configuração
copy .env.example .env

# Edite o .env e adicione sua OpenAI API Key
notepad .env
# Procure: OPENAI_API_KEY=your-openai-api-key-here
# Substitua por sua chave real
```

**⚠️ IMPORTANTE**: Se não tiver OpenAI key, pode pegar uma grátis em: https://platform.openai.com/api-keys

---

### **3️⃣ Instale as Dependências Python**

```powershell
# Crie ambiente virtual (recomendado)
python -m venv venv

# Ative o ambiente
.\venv\Scripts\Activate

# Instale dependências
pip install -r requirements.txt
```

**⏱️ Tempo estimado**: 5-8 minutos (PyTorch é pesado)

**💡 Dica**: Se der erro no PyTorch Geometric:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

---

### **4️⃣ Inicie o Neo4j (Docker)**

```powershell
# Inicie o container Neo4j
docker-compose up -d

# Aguarde Neo4j iniciar (30 segundos)
timeout /t 30

# Verifique se está rodando
docker ps
# Deve mostrar: neo4j:5.13-community
```

**🌐 Acesse o Neo4j Browser**: http://localhost:7474
- **Usuário**: neo4j
- **Senha**: midas123

---

### **5️⃣ Inicie o Serviço FastAPI**

```powershell
# Em um terminal separado
cd C:\Users\rugge_p2gkz2r\Desktop\midas-ai\midas-ai-service

# Ative o ambiente virtual
.\venv\Scripts\Activate

# Inicie o servidor
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**✅ Sucesso quando ver**:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**🌐 Acesse a API**: http://localhost:8000/docs

---

### **6️⃣ Popule o Grafo com Dados**

```powershell
# Em outro terminal (ou use a API docs)
curl -X POST http://localhost:8000/graph/populate

# Aguarde 15 segundos para população completar
timeout /t 15

# Verifique o status
curl http://localhost:8000/graph/stats
```

**📊 Deve retornar**:
```json
{
  "graph": {
    "nodes": 20,
    "edges": 40-80,
    "coherence": 0.6-0.8
  }
}
```

---

### **7️⃣ Treine a GNN (Opcional mas Recomendado)**

```powershell
# No terminal com venv ativo
python train_gnn.py
```

**⏱️ Tempo**: 2-3 minutos (CPU) ou 30 segundos (GPU)

**✅ Sucesso quando ver**:
```
Epoch 15/15 - Loss: 0.0567
✓ Model saved to: gnn_neuroelastic_pretrained.pt
```

---

### **8️⃣ Teste o Sistema!**

```powershell
# Teste 1: Query simples
curl -X POST http://localhost:8000/v2/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"Quanto gastei no ifood?\"}"

# Teste 2: What-if scenario
curl -X POST http://localhost:8000/v2/query ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"E se eu parar de pedir delivery?\"}"

# Teste 3: Health check
curl http://localhost:8000/health
```

---

## 🎬 DEMO AUTOMATIZADO (Mais Fácil!)

```powershell
# Execute o script de demo
python demo_mvp.py
```

Este script faz **TUDO** automaticamente:
1. ✅ Popula o grafo
2. ✅ Treina a GNN
3. ✅ Testa queries
4. ✅ Mostra métricas

---

## 🐛 TROUBLESHOOTING

### **Problema: Docker não inicia**
```powershell
# Solução 1: Reinicie o Docker Desktop
# Solução 2: Verifique se WSL2 está instalado
wsl --list --verbose
```

### **Problema: Neo4j não conecta**
```powershell
# Verifique logs do container
docker logs midas-ai-service_neo4j_1

# Reinicie o container
docker-compose down
docker-compose up -d
```

### **Problema: Erro ao instalar PyTorch**
```powershell
# Use versão CPU-only (mais leve)
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### **Problema: API retorna 500**
```powershell
# Verifique logs do servidor
# Terminal onde rodou uvicorn mostrará o erro

# Verifique se Neo4j está rodando
docker ps

# Teste conexão Neo4j
curl http://localhost:7474
```

---

## 📊 ENDPOINTS PRINCIPAIS

### **Health Check**
```bash
GET http://localhost:8000/health
```

### **Cognitive Query (FDR v2)**
```bash
POST http://localhost:8000/v2/query
Body: {"query": "sua pergunta aqui"}
```

### **Graph Stats**
```bash
GET http://localhost:8000/graph/stats
```

### **Populate Graph**
```bash
POST http://localhost:8000/graph/populate
```

### **Train GNN**
```bash
POST http://localhost:8000/gnn/train
```

---

## 🎯 VALIDAÇÃO FINAL

Execute estes comandos para confirmar que tudo está rodando:

```powershell
# 1. Docker
docker ps
# ✅ Deve mostrar container neo4j

# 2. Neo4j
curl http://localhost:7474
# ✅ Deve retornar página HTML

# 3. API
curl http://localhost:8000/health
# ✅ Deve retornar JSON com status: "healthy"

# 4. Grafo
curl http://localhost:8000/graph/stats
# ✅ Deve mostrar nodes > 0
```

---

## 🚀 PRONTO PARA APRESENTAR!

Agora você pode:
1. ✅ Fazer queries cognitivas
2. ✅ Visualizar ICE (Interface Cognitiva Estruturada)
3. ✅ Demonstrar raciocínio multi-hop
4. ✅ Mostrar GNN funcionando
5. ✅ Observar Aphelion Layer

**Documentação completa**: `SETUP_FDR_V2.md` e `MVP_PROOF_OF_CONCEPT.md`

**Dúvidas?** Todos os logs estão visíveis nos terminais! 🎉
