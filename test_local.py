import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.pipeline import RAGPipeline
from app.models.schemas import QueryRequest
from data.dummy.loader import DummyDataLoader


class LocalRAGTester:
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        self.dummy_loader = DummyDataLoader()
    
    async def setup(self):
        print("Configurando Midas AI Service para teste local...")
        
        # Setup dummy data
        self.dummy_loader.setup_dummy_environment()
        
        # Initialize RAG pipeline
        await self.rag_pipeline.initialize()
        
        print("Setup concluído!")
    
    async def test_queries(self):
        
        test_queries = [
            {
                "query": "Quanto gastei com delivery este mês?",
                "description": "Teste busca por categoria específica"
            },
            {
                "query": "Quais são minhas assinaturas ativas?",
                "description": "Teste busca por assinaturas"
            },
            {
                "query": "Qual é meu saldo total?",
                "description": "Teste consulta de saldo"
            },
            {
                "query": "Quanto gastei no total em outubro?",
                "description": "Teste agregação por período"
            },
            {
                "query": "Como estão minhas metas financeiras?",
                "description": "Teste consulta de cofrinhos"
            },
            {
                "query": "Gastei muito no iFood?",
                "description": "Teste busca por estabelecimento"
            },
            {
                "query": "Quais foram meus gastos com transporte?",
                "description": "Teste categoria transporte"
            },
            {
                "query": "Recebi meu salário este mês?",
                "description": "Teste busca por receitas"
            }
        ]
        
        print("\nTestando Queries do RAG...")
        print("=" * 60)
        
        for i, test in enumerate(test_queries, 1):
            print(f"\nTeste {i}: {test['description']}")
            print(f"Query: '{test['query']}'")
            print("-" * 40)
            
            try:
                # Process query
                result = await self.rag_pipeline.process_query(
                    query=test["query"],
                    user_id=123,
                    filters=None
                )
                
                # Display results
                print(f"Resposta: {result['answer']}")
                print(f"Tipo: {result['query_type']}")
                print(f"Confiança: {result['confidence']:.2f}")
                print(f"Tempo: {result['execution_time_ms']}ms")
                print(f"Resultados encontrados: {len(result['retrieval_results'])}")
                
                # Show top retrieval results
                if result['retrieval_results']:
                    print("Top resultados:")
                    for j, res in enumerate(result['retrieval_results'][:2], 1):
                        print(f"   {j}. [{res.source}] {res.content[:80]}... (score: {res.score:.3f})")
                
            except Exception as e:
                print(f"Erro: {str(e)}")
            
            print()
    
    async def interactive_mode(self):
        """Modo interativo para teste manual"""
        print("\nModo Interativo do RAG")
        print("Digite suas queries financeiras (ou 'quit' para sair):")
        print("=" * 50)
        
        while True:
            try:
                query = input("\n🔍 Query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'sair']:
                    print("Saindo do modo interativo...")
                    break
                
                if not query:
                    continue
            
                result = await self.rag_pipeline.process_query(
                    query=query,
                    user_id=123,
                    filters=None
                )
                
                # Display result
                print(f"\n{result['answer']}")
                print(f"Tipo: {result['query_type']} | Confiança: {result['confidence']:.2f} | {result['execution_time_ms']}ms")
                
            except KeyboardInterrupt:
                print("\n\nSaindo...")
                break
            except Exception as e:
                print(f"Erro: {str(e)}")
    
    def show_dummy_data_summary(self):
        """Mostra resumo dos dados dummy"""
        data = self.dummy_loader.load_all_data()

        print("\nResumo dos Dados Dummy:")
        print("=" * 40)
        
        # Transactions by category
        transactions = data.get("transactions", [])
        if transactions:
            categories = {}
            total_spent = 0
            
            for t in transactions:
                cat = t["category"]
                amount = abs(t["amount"]) if t["amount"] < 0 else 0
                
                if cat not in categories:
                    categories[cat] = {"count": 0, "total": 0}
                
                categories[cat]["count"] += 1
                categories[cat]["total"] += amount
                total_spent += amount
            
            print("Gastos por Categoria:")
            for cat, info in sorted(categories.items()):
                if info["total"] > 0:
                    print(f"   {cat}: {info['count']} transações - R$ {info['total']:.2f}")

            print(f"\nTotal Gasto: R$ {total_spent:.2f}")

        # Subscriptions
        subscriptions = data.get("subscriptions", [])
        active_subs = [s for s in subscriptions if s["active"]]
        
        if active_subs:
            print(f"\nAssinaturas Ativas ({len(active_subs)}):")
            total_monthly = 0
            for sub in active_subs:
                print(f"   {sub['name']}: R$ {sub['amount']:.2f}/mês")
                total_monthly += sub["amount"]
            print(f"   Total Mensal: R$ {total_monthly:.2f}")
        
        # Goals
        goals = data.get("goals", [])
        if goals:
            print(f"\nMetas Financeiras ({len(goals)}):")
            for goal in goals:
                progress = (goal["current_amount"] / goal["target_amount"]) * 100
                print(f"   {goal['name']}: R$ {goal['current_amount']:.0f} / R$ {goal['target_amount']:.0f} ({progress:.1f}%)")


async def main():
    """Função principal"""
    print("Midas AI Service - Teste Local")
    print("=" * 50)
    
    tester = LocalRAGTester()
    
    # Setup
    await tester.setup()
    
    # Show data summary
    tester.show_dummy_data_summary()
    
    # Run automated tests
    await tester.test_queries()
    
    # Interactive mode
    response = input("\nEntrar no modo interativo? (y/n): ").strip().lower()
    if response in ['y', 'yes', 's', 'sim']:
        await tester.interactive_mode()
    
    print("\nTeste concluído!")


if __name__ == "__main__":
    asyncio.run(main())