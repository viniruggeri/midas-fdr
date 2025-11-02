# Changelog

All notable changes to the MIDAS FDR project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2025-11-02

### 🎉 Major Release: FDR 2 - Deep Reasoning Layer

This release introduces a complete paradigm shift from traditional RAG to **persistent inferential reasoning** with neuroelastic graph topologies.

### Added

#### Core Framework
- **Neuroelastic Graph System** (`app/cognitive/neuroelastic_graph.py`)
  - Dynamic topological reasoning with Neo4j persistence
  - Semantic coherence calculation (cosine similarity of embeddings)
  - Multi-hop query execution with GNN refinement
  - Adaptive edge reweighting based on access patterns

- **GNN Reasoning Engine** (`app/cognitive/gnn_reasoner.py`)
  - Graph Attention Networks (GAT) with 2-layer architecture
  - Node relevance scoring and graph confidence prediction
  - Multi-hop reasoning with GNN-guided path discovery
  - Training pipeline on real Neo4j data

- **Aphelion Layer** (`app/cognitive/aphelion.py`)
  - Semantic survival mechanism with formal mathematical foundation
  - Extinction/rebirth cycles for contextual homeostasis
  - PageRank-based core concept extraction (Neo4j GDS)
  - Detailed metrics tracking (coherence gain, pruned nodes/edges)

- **Multi-hop Reasoning Engine** (`app/cognitive/reasoning_engine.py`)
  - ICE (Instruction-Context-Examples) prompt assembly
  - GNN-enhanced confidence boosting (0.75 → 0.97)
  - Iterative depth-first search with relevance tracking
  - Operation type classification (simple/pattern/what-if)

#### Documentation
- **Comprehensive Whitepaper** (`whitepaper-fdr.md`)
  - 20-section formal paper with mathematical notation
  - Core concepts: neuroelasticity, topology, entropy regulation
  - Stack recommendations and architecture diagrams
  - References and appendix with formulas

- **Quick Start Guide** (`QUICK_START.md`)
  - 8-step setup process (~15 minutes)
  - Troubleshooting section for common issues
  - Endpoint documentation with curl examples
  - Validation checklist

- **MVP Proof of Concept** (`MVP_PROOF_OF_CONCEPT.md`)
  - Evidence of functional GNN reasoning
  - Comparison: PoC vs Production requirements
  - Demo script and testing instructions

- **Setup Automation**
  - PowerShell script (`setup.ps1`) for Windows
  - Bash script (`setup.sh`) for Linux/Mac
  - Docker detection and interactive configuration

#### Training & Testing
- **GNN Training Script** (`train_gnn.py`)
  - Fetches training data from Neo4j
  - Supervised learning on access_count (15 epochs)
  - Model serialization (gnn_neuroelastic_pretrained.pt)
  - Test inference validation

- **Automated Demo** (`demo_mvp.py`)
  - End-to-end system testing
  - Tests simple/pattern/what-if queries
  - Displays metrics and GNN enhancement status

#### API Enhancements
- **New Endpoints:**
  - `POST /gnn/train` - Trigger GNN training
  - Enhanced `/graph/stats` with GNN status
  - Enhanced `/graph/populate` with 20 transactions

- **Enhanced Health Check:**
  - Neo4j connection status
  - GNN model loaded status
  - Graph statistics (nodes, edges, coherence)

### Changed

- **Requirements:**
  - Added `torch>=2.1.0`
  - Added `torch-geometric>=2.4.0`
  - Added `torch-scatter>=2.1.2`
  - Added `torch-sparse>=0.6.18`

- **Graph Schema:**
  - Enhanced with `access_count` property for training
  - Added `embedding` property for semantic coherence
  - Optimized indexing for multi-hop queries

- **README.md:**
  - Rewritten with FDR 2 concepts
  - Added badges and quick links
  - Core concepts section with mathematical formulas
  - Updated architecture diagram

### Fixed

- **Setup Script Issues:**
  - Fixed `.venv` vs `venv` detection
  - Fixed emoji encoding errors in PowerShell
  - Added fallback `.env` creation when `.env.example` missing
  - Improved error handling and user feedback

- **Git Issues:**
  - Resolved divergent branch with `git pull --rebase`
  - Updated `.gitignore` for `.venv` and model files

### Technical Details

**Architecture:**
- Neo4j 5.13-community with APOC + GDS plugins
- PyTorch Geometric 2.4.0+ for GNN operations
- FastAPI 0.104+ for async API
- SentenceTransformers for embeddings

**Performance:**
- Simple query: ~100-200ms
- Pattern detection: ~300-500ms
- GNN-enhanced: ~500-1000ms
- Training: 2-3 min CPU, 20-30s GPU

**Model Specs:**
- GAT layers: 2 (input→hidden→output)
- Attention heads: 4 (layer 1) → 1 (layer 2)
- Parameters: ~50,000
- Model size: ~200KB

---

## [1.0.0] - 2025-10-15

### Initial Release

- Basic RAG implementation with FastAPI
- Neo4j integration for transaction storage
- Simple query processing
- HumanizerLLM for response generation

---

**Legend:**
- 🎉 Major release
- ✨ Feature
- 🐛 Bug fix
- 📝 Documentation
- ⚡ Performance
- 🔒 Security
