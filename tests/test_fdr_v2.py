"""
MIDAS FDR v2 - Test Script
Testes E2E do sistema de raciocínio cognitivo
"""

import asyncio
import json
from app.cognitive import NeuroelasticGraph, MIDASCognitiveEngine, HumanizerLLM, AphelionLayer


async def test_fdr_v2():
    print("=" * 80)
    print("MIDAS FDR v2 - Test Suite")
    print("=" * 80)
    
    # Inicializar componentes
    print("\n1. Inicializando componentes...")
    graph = NeuroelasticGraph()
    aphelion = AphelionLayer(graph)
    cognitive_engine = MIDASCognitiveEngine(graph, aphelion)
    humanizer = HumanizerLLM()
    
    # Popular grafo com dados dummy
    print("\n2. Populando grafo com transações dummy...")
    dummy_transactions = [
        {"id": "tx001", "merchant": "ifood", "category": "alimentação", "amount": 45.0, "description": "delivery almoço", "timestamp": "2024-01-15T12:00:00"},
        {"id": "tx002", "merchant": "uber", "category": "transporte", "amount": 18.5, "description": "corrida centro", "timestamp": "2024-01-15T14:30:00"},
        {"id": "tx003", "merchant": "ifood", "category": "alimentação", "amount": 52.0, "description": "delivery jantar", "timestamp": "2024-01-16T20:00:00"},
        {"id": "tx004", "merchant": "rappi", "category": "alimentação", "amount": 38.0, "description": "delivery lanche", "timestamp": "2024-01-17T16:00:00"},
        {"id": "tx005", "merchant": "uber", "category": "transporte", "amount": 22.0, "description": "corrida aeroporto", "timestamp": "2024-01-18T08:00:00"},
    ]
    
    for tx in dummy_transactions:
        await graph.add_transaction_node(**tx)
    
    print(f"   ✓ Adicionadas {len(dummy_transactions)} transações")
    
    # Estatísticas iniciais
    print("\n3. Estatísticas do grafo...")
    stats = await graph.get_stats()
    print(f"   Nós: {stats['nodes']}")
    print(f"   Arestas: {stats['edges']}")
    print(f"   Coerência: {stats['coherence']:.4f}")
    
    # Test Case 1: Simple Query
    print("\n4. Test Case 1: Simple Query")
    print("   Query: 'Quanto gastei no ifood?'")
    
    ice_output = await cognitive_engine.reason("Quanto gastei no ifood?")
    print(f"   Tipo de operação: {ice_output.operation_type}")
    print(f"   Contexto ativado: {len(ice_output.context_activated)} nós")
    print(f"   Passos de inferência: {len(ice_output.inference_chain)}")
    print(f"   Confiança: {ice_output.final_conclusion.confidence_score:.2f}")
    
    humanized = await humanizer.humanize(ice_output)
    print(f"\n   Resposta humanizada:\n   {humanized[:200]}...")
    
    # Test Case 2: What-If Scenario
    print("\n5. Test Case 2: What-If Scenario")
    print("   Query: 'E se eu parar de pedir ifood por um mês?'")
    
    ice_output = await cognitive_engine.reason("E se eu parar de pedir ifood por um mês?")
    print(f"   Tipo de operação: {ice_output.operation_type}")
    print(f"   Contexto ativado: {len(ice_output.context_activated)} nós")
    print(f"   Passos de inferência: {len(ice_output.inference_chain)}")
    
    for step in ice_output.inference_chain:
        print(f"   Step {step.step_number}: {step.operation} - {step.intermediate_result}")
    
    # Test Case 3: Multi-hop Query
    print("\n6. Test Case 3: Multi-hop Query")
    if ice_output.context_activated:
        start_node = ice_output.context_activated[0].node_id
        print(f"   Starting from node: {start_node}")
        
        results = await graph.multi_hop_query(start_node, max_depth=2, max_results=5)
        print(f"   Multi-hop results: {len(results)} nós alcançados")
        for r in results[:3]:
            print(f"      - {r['node_id']}: distance={r['distance']:.2f}")
    
    # Test Case 4: Aphelion Layer
    print("\n7. Test Case 4: Aphelion Layer (Survival Check)")
    survival = await aphelion.check_survival()
    print(f"   Status: {survival['status']}")
    print(f"   Coerência: {survival['coherence']:.4f}")
    print(f"   Ação: {survival['action']}")
    
    # Estatísticas finais
    print("\n8. Estatísticas finais...")
    stats = await graph.get_stats()
    print(f"   Nós: {stats['nodes']}")
    print(f"   Arestas: {stats['edges']}")
    print(f"   Coerência: {stats['coherence']:.4f}")
    
    # Salvar exemplo de ICE output
    print("\n9. Salvando ICE output example...")
    with open("ice_output_example.json", "w", encoding="utf-8") as f:
        f.write(ice_output.to_json())
    print("   ✓ Salvo em ice_output_example.json")
    
    # Cleanup
    await graph.close()
    
    print("\n" + "=" * 80)
    print("✓ Test Suite Completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_fdr_v2())
