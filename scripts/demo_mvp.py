"""
MIDAS FDR v2 - Quick Demo Script
Demonstração rápida do MVP com GNN
"""

import asyncio
import requests
import time
import json

BASE_URL = "http://localhost:8000"


def print_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def demo():
    print_section("🚀 MIDAS FDR v2 - MVP Demo")
    
    # 1. Health check
    print("1️⃣  Health Check...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        health = r.json()
        print(f"   Status: {health['status']}")
        if 'fdr_v2' in health.get('checks', {}):
            fdr_stats = health['checks']['fdr_v2']
            print(f"   Graph: {fdr_stats['graph_nodes']} nodes, {fdr_stats['graph_edges']} edges")
            print(f"   Coherence: {fdr_stats['coherence']:.3f}")
    except Exception as e:
        print(f"   ❌ Service not running. Start with: uvicorn app.main:app --reload")
        print(f"   Error: {e}")
        return
    
    # 2. Populate graph
    print_section("2️⃣  Populating Graph")
    print("   Sending 20 transactions to Neo4j...")
    r = requests.post(f"{BASE_URL}/graph/populate")
    print(f"   {r.json()['message']}")
    print("   ⏳ Waiting 15 seconds for population...")
    time.sleep(15)
    
    # 3. Check graph stats
    print_section("3️⃣  Graph Statistics")
    r = requests.get(f"{BASE_URL}/graph/stats")
    stats = r.json()
    print(f"   Nodes: {stats['graph']['nodes']}")
    print(f"   Edges: {stats['graph']['edges']}")
    print(f"   Coherence: {stats['graph']['coherence']:.4f}")
    print(f"   GNN Enabled: {stats['gnn']['enabled']}")
    print(f"   GNN Loaded: {stats['gnn']['model_loaded']}")
    
    # 4. Train GNN
    print_section("4️⃣  Training GNN")
    print("   Starting GNN training...")
    print("   (This will take 2-3 minutes)")
    print("\n   Run in separate terminal: python train_gnn.py")
    print("   OR trigger via API: POST /gnn/train")
    input("\n   Press ENTER after training completes...")
    
    # 5. Test queries
    print_section("5️⃣  Cognitive Query Test")
    
    queries = [
        "Quanto gastei no ifood?",
        "E se eu parar de pedir delivery?",
        "Qual é o padrão de gastos com transporte?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n   Query {i}: '{query}'")
        print("   " + "-" * 70)
        
        start = time.time()
        r = requests.post(
            f"{BASE_URL}/v2/query",
            json={"query": query},
            timeout=30
        )
        duration = time.time() - start
        
        result = r.json()
        
        print(f"   Operation: {result['operation_type']}")
        print(f"   Context nodes: {len(result['context_activated'])}")
        print(f"   Inference steps: {len(result['inference_chain'])}")
        
        if result['inference_chain']:
            last_step = result['inference_chain'][-1]
            print(f"   Last step: {last_step['operation']}")
            print(f"   Confidence: {last_step['confidence']:.2f}")
            if 'GNN' in last_step['description']:
                print("   ✅ GNN-ENHANCED!")
        
        print(f"   Duration: {duration:.2f}s")
        
        if result.get('humanized_response'):
            response_preview = result['humanized_response'][:150] + "..."
            print(f"\n   Response preview:\n   {response_preview}")
    
    # 6. Final stats
    print_section("6️⃣  Final Graph State")
    r = requests.get(f"{BASE_URL}/graph/stats")
    stats = r.json()
    print(f"   Nodes: {stats['graph']['nodes']}")
    print(f"   Edges: {stats['graph']['edges']}")
    print(f"   Coherence: {stats['graph']['coherence']:.4f}")
    print(f"   Extinctions: {stats['aphelion'].get('total_extinctions', 0)}")
    
    print_section("✅ Demo Completed!")
    print("📊 Key Metrics:")
    print("   • GNN trained on real graph data")
    print("   • Multi-hop reasoning with neural attention")
    print("   • ICE with explicit inference chain")
    print("   • Neuroelastic adaptation active")
    print("\n📖 See MVP_PROOF_OF_CONCEPT.md for details")


if __name__ == "__main__":
    asyncio.run(demo())
