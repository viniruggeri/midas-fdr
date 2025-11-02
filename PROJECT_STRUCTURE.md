# Project Structure

```
midas-ai-service/
├── 📁 app/                         # Core application code
│   ├── cognitive/                  # Reasoning engines
│   │   ├── neuroelastic_graph.py   # Graph topology manager
│   │   ├── gnn_reasoner.py         # GNN inference engine
│   │   ├── aphelion.py             # Extinction/rebirth layer
│   │   └── reasoning_engine.py     # Multi-hop orchestrator
│   ├── services/                   # Business logic services
│   └── main.py                     # FastAPI application
│
├── 📁 docs/                        # Documentation
│   ├── whitepaper-fdr.md           # Academic whitepaper (20 sections)
│   ├── QUICK_START.md              # Setup guide (15 minutes)
│   ├── MVP_PROOF_OF_CONCEPT.md     # Implementation evidence
│   └── CHANGELOG.md                # Version history
│
├── 📁 scripts/                     # Executable scripts
│   ├── setup.ps1                   # Automated setup (Windows)
│   ├── train_gnn.py                # GNN training pipeline
│   └── demo_mvp.py                 # End-to-end demo
│
├── 📁 tests/                       # Test files
│   ├── test_fdr_v2.py              # FDR v2 integration tests
│   ├── test_improvements.py        # Feature tests
│   └── test_local.py               # Local development tests
│
├── 📁 data/                        # Data storage
│   └── (Neo4j data, model checkpoints)
│
├── 📄 README.md                    # Project overview
├── 📄 requirements.txt             # Python dependencies
├── 📄 docker-compose.yml           # Neo4j container config
├── 📄 config.py                    # Application configuration
├── 📄 .env                         # Environment variables (not in git)
├── 📄 .env.example                 # Template for .env
└── 📄 LICENSE                      # MIT License

```

## Quick Navigation

### 🚀 Getting Started
1. **Setup:** `.\scripts\setup.ps1` (Windows) or see `docs/QUICK_START.md`
2. **Train GNN:** `python scripts/train_gnn.py`
3. **Run Demo:** `python scripts/demo_mvp.py`

### 📚 Documentation
- **Theory:** [docs/whitepaper-fdr.md](docs/whitepaper-fdr.md)
- **Implementation:** [docs/MVP_PROOF_OF_CONCEPT.md](docs/MVP_PROOF_OF_CONCEPT.md)
- **Changes:** [docs/CHANGELOG.md](docs/CHANGELOG.md)

### 💻 Development
- **Main App:** `app/main.py` (FastAPI endpoints)
- **Cognitive Core:** `app/cognitive/` (reasoning engines)
- **Tests:** `tests/` (pytest compatible)

### 🐳 Services
- **Neo4j:** `docker-compose up -d` (http://localhost:7474)
- **FastAPI:** `uvicorn app.main:app --reload` (http://localhost:8000)

## File Count

- **Python files:** ~15
- **Documentation:** 5 markdown files
- **Scripts:** 3 executable files
- **Tests:** 3 test suites
- **Total lines of code:** ~2,687 (Python only)

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- FastAPI 0.104+
- Neo4j 5.13
- PyTorch 2.1+
- PyTorch Geometric 2.4+
- SentenceTransformers

## Clean Organization

✅ **docs/** — All documentation in one place  
✅ **scripts/** — All executable scripts  
✅ **tests/** — All test files  
✅ **app/** — Core application logic  
✅ Root level — Only essential config files

---

**Last updated:** November 2, 2025  
**Version:** 2.0.0
