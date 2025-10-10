# 🧪 Dados Dummy para Teste Local

Este diretório contém dados de teste (dummy data) para testar o **RAG do Midas AI Service** localmente, sem precisar de banco de dados Oracle ou PostgreSQL.

## 📊 Dados Incluídos

### 1. `transactions.json` - Transações Financeiras
- **15 transações** do usuário ID 123
- Período: Setembro-Outubro 2025
- Categorias: delivery, assinaturas, salário, transporte, alimentação, saúde, lazer
- Bancos: Nubank, Itaú
- Valores variados de R$ 9,90 a R$ 4.500,00

### 2. `subscriptions.json` - Assinaturas Ativas
- **6 assinaturas** (5 ativas, 1 cancelada)
- Netflix, Spotify, Amazon Prime, Disney Plus, Gympass, Adobe
- Total mensal ativo: ~R$ 116,50

### 3. `accounts.json` - Contas Bancárias
- **3 contas**: 2 Nubank, 1 Itaú
- Tipos: conta corrente e poupança
- Saldo total: R$ 5.590,25

### 4. `goals.json` - Metas Financeiras (Cofrinhos)
- **4 metas**: Viagem Disney, iPhone 16, Reserva de Emergência, Carro Novo
- Valores de R$ 5.500 a R$ 45.000
- Progresso variado (15% a 80%)

### 5. `documents_for_rag.json` - Documentos para RAG
- **15 documentos** textuais processados
- Cada transação convertida em texto natural
- Metadados completos para retrieval
- Pronto para embeddings FAISS e TF-IDF

## 🧪 Como Testar

### 1. Setup Inicial
```bash
# Windows
setup_windows.bat

# Linux/Mac  
./setup.sh
```

### 2. Teste Automático
```bash
# Windows
test_dummy.bat

# Linux/Mac
python test_local.py
```

### 3. Teste Manual
```python
python test_local.py
# Escolha modo interativo quando perguntado
```

## 🔍 Queries de Teste Sugeridas

### Gastos por Categoria
- ✅ "Quanto gastei com delivery este mês?"
- ✅ "Quais foram meus gastos com transporte?"
- ✅ "Gastei muito no iFood?"

### Assinaturas
- ✅ "Quais são minhas assinaturas ativas?"
- ✅ "Quanto pago por mês em streaming?"
- ✅ "Tenho Netflix ativo?"

### Saldos e Totais
- ✅ "Qual é meu saldo total?"
- ✅ "Quanto gastei no total em outubro?"
- ✅ "Recebi meu salário este mês?"

### Metas Financeiras
- ✅ "Como estão meus cofrinhos?"
- ✅ "Quanto falta para minha viagem?"
- ✅ "Minhas metas financeiras estão no prazo?"

### Queries Abertas
- ✅ "Para onde foi meu dinheiro?"
- ✅ "Estou gastando muito?"
- ✅ "Como está minha situação financeira?"

## 📈 Dados Estatísticos

- 👤 **User ID**: 123
- 💳 **Transações**: 15
- 📺 **Assinaturas**: 6 (5 ativas)
- 🏦 **Contas**: 3
- 🎯 **Metas**: 4
- 💰 **Saldo Total**: R$ 5.590,25
- 💸 **Total Gasto**: R$ 767,90
- 💵 **Total Recebido**: R$ 4.500,00

### Gastos por Categoria:
- **delivery**: 3 transações - R$ 86,40
- **assinaturas**: 4 transações - R$ 80,60  
- **transporte**: 2 transações - R$ 135,50
- **alimentação**: 2 transações - R$ 197,80
- **saúde**: 1 transação - R$ 45,80
- **lazer**: 1 transação - R$ 32,00
- **transferência**: 1 transação - R$ 200,00

## 🏛️ Integração com RAG

Os dados dummy são automaticamente:

1. **Carregados** pelo `DummyDataLoader`
2. **Indexados** no FAISS (embeddings semânticos)
3. **Indexados** no TF-IDF (busca sintática)
4. **Processados** pelo pipeline RAG híbrido
5. **Formatados** em respostas naturais

### Fluxo de Teste:
```
Dummy Data → RAG Pipeline → Query Processing → Natural Response
```

O sistema simula perfeitamente o comportamento do RAG em produção, mas usando dados locais estáticos.

---

🧪 **Perfeito para desenvolvimento e demonstrações do Midas AI Service!**