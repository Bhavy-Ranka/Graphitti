# Graphitti

Graph-native web intelligence: crawl → chunk + extract triples (local NLTK
NLP pipeline, no LLM calls) → consistency check → Neo4j knowledge graph →
LangChain query routing → multi-strategy retrieval (dense / BM25 / graph /
hybrid RRF) → LangGraph multi-agent orchestration → cited answer →
Streamlit dashboard.

<img width="1693" height="1129" alt="Screenshot from 2026-07-21 14-36-12" src="https://github.com/user-attachments/assets/aeb3dca1-c12b-4dc6-a6b2-42783135ce4b" />
<img width="1693" height="1129" alt="Screenshot from 2026-07-21 14-38-12" src="https://github.com/user-attachments/assets/8286d838-76f5-432d-ad86-99759504443d" />
<img width="1693" height="1129" alt="Screenshot from 2026-07-21 15-12-18" src="https://github.com/user-attachments/assets/7b33a11c-063d-4bbc-8d4d-d4b3fd203504" />

## Project Structure

```text
graphitti/
├── config.py                # Centralized environment settings (.env)
├── crawling/
│   └── crawler.py           # Playwright BFS web crawler
├── extraction/
│   ├── chunking.py          # Sentence-boundary semantic chunking
│   ├── schemas.py           # Pydantic schemas for extracted triples
│   ├── nlp_pipeline.py      # Local NLTK extraction pipeline (POS, NER, SVO)
│   ├── triple_extractor.py  # Backend dispatcher (NLTK vs. Groq LLM)
│   ├── direction.py         # Subject/Object direction normalization
│   └── entity_resolution.py # Entity alias resolution & merging
├── graph/
│   ├── store.py             # Neo4j & In-Memory GraphStore implementations
│   ├── loader.py            # Neo4j UNWIND/MERGE batch loader with rollback
│   └── consistency.py       # Graph conflict detection & contested edge tagging
├── retrieval/
│   ├── text_index.py        # Addressable chunk index for raw page text
│   ├── bm25.py              # Sparse BM25 retrieval strategy
│   ├── dense.py             # Vector search (Sentence-Transformers / TF-IDF)
│   ├── graph_retrieval.py   # Multi-hop & entity-centric graph traversal
│   └── base.py              # HybridFusionStrategy (RRF) & retrieval registry
├── routing/
│   ├── schemas.py           # Intent classification & query decomposition schemas
│   ├── chains.py            # LCEL chains for classification and rephrasing
│   └── query_router.py      # Main router with heuristic fallback support
├── orchestration/
│   ├── orchestrator.py      # LangGraph state machine (Routing → Retrieval → Synthesis)
│   └── synthesis_chain.py   # Final answer generation chain with provenance
├── api/
│   ├── app.py               # FastAPI REST endpoints
│   └── static/graph.html    # Interactive graph visualizer (vis-network)
├── cli/
│   └── main.py              # CLI tool for crawling, extracting, and loading
├── evaluation/
│   └── ablation.py          # Precision/Recall/Latency benchmarking suite
└── streamlit_app.py         # Streamlit interactive UI dashboard
```

## One graph at a time

`POST /ingest` defaults to `reset=true`: every crawl clears the existing
Neo4j graph and text index before loading the new one, so ingesting a new
URL always replaces the previous entity's graph instead of merging into it.
The Streamlit dashboard always calls `/ingest` with `reset=true` and
re-fetches `/graph` after every ingest, so the graph panel only ever shows
the entity you just crawled.

## Retrieval strategies

- `sparse_bm25` — BM25Okapi over chunk text (`retrieval/bm25.py`)
- `dense_vector` — sentence-transformers if available, TF-IDF + cosine
  fallback otherwise (`retrieval/dense.py`)
- `graph_traversal` — fuzzy-match query entities, multi-hop BFS
  (`GRAPH_MAX_HOPS`, default 2)
- `graph_1hop` — same traversal strategy restricted to 1 hop, used for
  `single_fact_lookup` intents
- `entity_centric` — full 1-hop neighborhood of the best-matching entity,
  sorted by edge confidence
- `hybrid_fusion` — reciprocal-rank fusion across all of the above

`routing/query_router.py` maps each classified intent to one of these
strategies (`STRATEGY_MAP`), falling back to `hybrid_fusion` whenever
intent-classification confidence is below `confidence_floor` (0.55).

## Extraction backend

Triple extraction defaults to a **local NLTK pipeline** (`EXTRACTION_BACKEND=nltk`,
or just leave it unset) — POS tagging, named-entity recognition, a shallow
NP/VP/PP grammar, and rule-based SVO / passive-voice / prepositional-phrase
triple extraction, with lightweight pronoun resolution and heuristic
confidence scoring. It runs entirely on your machine: no API calls, no
per-request latency, no rate limits, and it works with no `GROQ_API_KEY` at
all. The first run downloads a few small NLTK corpora automatically (cached
afterwards).

Set `EXTRACTION_BACKEND=groq` in `.env` to use the original LangChain +
ChatGroq LLM extractor instead (higher quality on ambiguous/complex
sentences, at the cost of API latency and Groq's rate limits). Query
routing (`routing/chains.py`) and orchestration still use Groq either way —
this switch only affects triple extraction.

## Running it

```bash
pip install -r requirements.txt --break-system-packages   # or a venv, drop the flag
playwright install chromium
cp .env.example .env   # NEO4J_PASSWORD required; GROQ_API_KEY only needed for
                        # query routing/orchestration, or EXTRACTION_BACKEND=groq

# 1. Backend API (needs a running Neo4j instance; GROQ_API_KEY optional unless
#    you want LLM-backed query routing/orchestration)
uvicorn graphitti.api.app:app --reload

# 2. Streamlit dashboard, in a second terminal
streamlit run graphitti/streamlit_app.py
```

Open the Streamlit URL it prints, paste a URL in the sidebar, click
**Crawl & Ingest**, then use the **Knowledge Graph** tab to see that
entity's graph and the **Ask a Question** tab to query it.
