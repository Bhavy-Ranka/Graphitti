from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from graphitti.config import MAX_DEPTH, MAX_PAGES
from graphitti.crawling.crawler import crawl_sync
from graphitti.extraction.triple_extractor import extract_triples_from_pages
from graphitti.graph.consistency import mark_contested
from graphitti.graph.loader import GraphLoader
from graphitti.graph.store import get_graph_store
from graphitti.orchestration.orchestrator import Orchestrator
from graphitti.retrieval.text_index import TextChunkStore
from graphitti.routing.query_router import QueryRouter

app = FastAPI(title="Graphitti", description="Graph-native web intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).parent / "static"

_state = {
    "text_store": TextChunkStore(),
    "graph_store": get_graph_store(),
    "router": QueryRouter(),
    "spin_up_seconds": None,
    "last_ingest_url": None,
}


def _orchestrator() -> Orchestrator:
    return Orchestrator(_state["text_store"], _state["graph_store"], router=_state["router"])


class IngestRequest(BaseModel):
    url: str
    depth: int = MAX_DEPTH
    pages: int = MAX_PAGES
    reset: bool = True


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    t0 = time.time()
    pages = crawl_sync(req.url, max_depth=req.depth, max_pages=req.pages)
    if not pages:
        raise HTTPException(400, "crawl returned no pages")

    if req.reset:
        _state["text_store"].clear()
        loader = GraphLoader()
        try:
            loader.clear_all()
        finally:
            loader.close()

    _state["text_store"].add_pages(pages)

    triples = extract_triples_from_pages(pages)
    triples, conflicts = mark_contested(triples)

    loader = GraphLoader()
    try:
        batch_id = loader.load_triples(triples)
    finally:
        loader.close()

    elapsed = time.time() - t0
    _state["spin_up_seconds"] = elapsed
    _state["last_ingest_url"] = req.url
    return {
        "pages_crawled": len(pages),
        "chunks_indexed": len(_state["text_store"]),
        "triples_extracted": len(triples),
        "contested_claim_groups": len(conflicts),
        "batch_id": batch_id,
        "reset": req.reset,
        "spin_up_seconds": round(elapsed, 2),
        "source_url": req.url,
    }


@app.post("/reset")
def reset():
    _state["text_store"].clear()
    loader = GraphLoader()
    try:
        loader.clear_all()
    finally:
        loader.close()
    _state["last_ingest_url"] = None
    return {"status": "cleared"}


@app.post("/query")
def query(req: QueryRequest):
    result = _orchestrator().answer(req.query)
    routing = result.get("routing")
    return {
        "query": result.get("query"),
        "answer": result.get("answer"),
        "intent": routing["intent"] if routing else None,
        "intent_confidence": routing["intent_confidence"] if routing else None,
        "rephrased_query": routing["rephrased_query"] if routing else None,
        "strategy_used": result.get("strategy_used"),
        "retries": result.get("retries"),
        "faithfulness": result.get("faithfulness"),
        "terminated_reason": result.get("terminated_reason"),
        "contested_warnings": result.get("contested_warnings"),
        "evidence": result.get("evidence"),
        "trace": result.get("trace"),
    }


@app.get("/graph")
def graph():
    gs = _state["graph_store"]
    with gs.driver.session() as session:
        nodes = [dict(r) for r in session.run("MATCH (e:Entity) RETURN e.name AS id, e.type AS type")]
        edges = [dict(r) for r in session.run(
            "MATCH (s:Entity)-[r:REL]->(o:Entity) "
            "RETURN s.name AS source, o.name AS target, r.predicate AS predicate, r.contested AS contested"
        )]
    return {"nodes": nodes, "edges": edges, "source_url": _state["last_ingest_url"]}


@app.get("/graph/view", response_class=HTMLResponse)
def graph_view():
    return (_STATIC_DIR / "graph.html").read_text()
