from __future__ import annotations

import time
from typing import Any, Optional, TypedDict

from langgraph.graph import END, StateGraph

from graphitti.config import ORCHESTRATOR_LATENCY_CEILING_S, ORCHESTRATOR_MAX_RETRIES, RETRIEVAL_TOP_K
from graphitti.graph.store import GraphStore
from graphitti.orchestration.synthesis_chain import synthesize_answer # type: ignore
from graphitti.retrieval.base import build_strategy_registry
from graphitti.retrieval.text_index import TextChunkStore
from graphitti.routing.query_router import QueryRouter


class OrchestratorState(TypedDict, total=False):
    query: str
    trace: list
    start_time: float
    routing: Optional[dict]
    strategy_used: Optional[str]
    retrieved_ids: list
    evidence: list
    answer: Optional[str]
    faithfulness: Optional[float]
    contested_warnings: list
    retries: int
    terminated_reason: Optional[str]


def _log(state: OrchestratorState, agent: str, note: str) -> None:
    state.setdefault("trace", []).append({"agent": agent, "note": note, "t": round(time.time(), 3)})


class Orchestrator:
    def __init__(self, text_store: TextChunkStore, graph_store: GraphStore,
                 router: QueryRouter | None = None, max_retries: int = ORCHESTRATOR_MAX_RETRIES):
        self.text_store = text_store
        self.graph_store = graph_store
        self.router = router or QueryRouter()
        self.strategies = build_strategy_registry(text_store, graph_store)
        self.max_retries = max_retries
        self._graph = self._build_graph()


    def _routing_node(self, state: OrchestratorState) -> dict:
        routing = self.router.route(state["query"])
        _log(state, "RoutingAgent", f"intent={routing['intent']} strategy={routing['chosen_strategy']}")
        return {"routing": routing, "strategy_used": routing["chosen_strategy"]}

    def _retrieval_node(self, state: OrchestratorState) -> dict:
        strategy = self.strategies.get(state["strategy_used"], self.strategies["hybrid_fusion"])
        routing = state.get("routing")
        query_for_retrieval = routing["rephrased_query"] if routing else state["query"]
        ids, latency_ms = strategy.timed_retrieve(query_for_retrieval, RETRIEVAL_TOP_K)
        evidence = self._resolve_evidence(ids)
        _log(state, strategy.name, f"retrieved {len(ids)} items in {latency_ms:.1f}ms")
        return {"retrieved_ids": ids, "evidence": evidence}

    def _resolve_evidence(self, ids: list[str]) -> list[dict]:
        evidence = []
        chunk_lookup = dict(zip(self.text_store.ids, self.text_store.texts))
        for cid in ids:
            if cid in chunk_lookup:
                meta = self.text_store.meta.get(cid, {})
                evidence.append({"id": cid, "text": chunk_lookup[cid], "source_url": meta.get("url", "")})
            else:
                neighbors = self.graph_store.neighbors(cid)
                lines = [
                    f"{cid} {n['predicate']} {n['neighbor']}" if n["direction"] == "out"
                    else f"{n['neighbor']} {n['predicate']} {cid}"
                    for n in neighbors[:5]
                ]
                summary = "; ".join(lines)
                src = neighbors[0]["source_url"] if neighbors else ""
                evidence.append({"id": cid, "text": summary or cid, "source_url": src})
        return evidence

    def _synthesis_node(self, state: OrchestratorState) -> dict:
        evidence = state.get("evidence", [])
        answer = synthesize_answer(state["query"], evidence[:RETRIEVAL_TOP_K])
        _log(state, "SynthesisAgent", f"drafted answer from {len(evidence)} evidence items")
        return {"answer": answer}

    def _faithfulness_node(self, state: OrchestratorState) -> dict:
        evidence = state.get("evidence", [])
        if not evidence:
            _log(state, "FaithfulnessAgent", "grounding_overlap=0.0")
            return {"faithfulness": 0.0}
        evidence_words = set(" ".join(e["text"] for e in evidence).lower().split())
        answer_words = set((state.get("answer") or "").lower().split())
        overlap = len(answer_words & evidence_words) / max(len(answer_words), 1)
        faithfulness = round(overlap, 3)
        _log(state, "FaithfulnessAgent", f"grounding_overlap={faithfulness}")
        return {"faithfulness": faithfulness}

    def _replan_node(self, state: OrchestratorState) -> dict:
        retries = state.get("retries", 0) + 1
        _log(state, "Orchestrator", f"re-planning (retry {retries}): switching to hybrid_fusion")
        return {"retries": retries, "strategy_used": "hybrid_fusion"}

    def _integrity_node(self, state: OrchestratorState) -> dict:
        contested = self.graph_store.contested_edges()
        touched = {e["id"] for e in state.get("evidence", [])}
        warnings = [c for c in contested if c.get("subject") in touched or c.get("object") in touched]
        if warnings:
            _log(state, "IntegrityAgent", f"{len(warnings)} contested claim(s) touch this answer")
        elapsed_ms = (time.time() - state["start_time"]) * 1000
        _log(state, "Orchestrator", f"terminated: {state.get('terminated_reason')}, total_latency_ms={elapsed_ms:.1f}")
        return {"contested_warnings": warnings}


    def _check_termination_node(self, state: OrchestratorState) -> dict:
        elapsed = time.time() - state["start_time"]
        if elapsed > ORCHESTRATOR_LATENCY_CEILING_S:
            return {"terminated_reason": "latency_ceiling_reached"}
        if (state.get("faithfulness") or 0.0) >= 0.3:
            return {"terminated_reason": "faithfulness_ok"}
        if state.get("retries", 0) >= self.max_retries:
            return {"terminated_reason": "max_retries_exhausted"}
        return {"terminated_reason": None}

    def _route_after_check(self, state: OrchestratorState) -> str:
        return "integrity" if state.get("terminated_reason") else "replan"


    def _build_graph(self):
        g = StateGraph(OrchestratorState)
        g.add_node("routing", self._routing_node)
        g.add_node("retrieval", self._retrieval_node)
        g.add_node("synthesis", self._synthesis_node)
        g.add_node("faithfulness", self._faithfulness_node)
        g.add_node("check_termination", self._check_termination_node)
        g.add_node("replan", self._replan_node)
        g.add_node("integrity", self._integrity_node)

        g.set_entry_point("routing")
        g.add_edge("routing", "retrieval")
        g.add_edge("retrieval", "synthesis")
        g.add_edge("synthesis", "faithfulness")
        g.add_edge("faithfulness", "check_termination")
        g.add_conditional_edges(
            "check_termination", self._route_after_check, {"replan": "replan", "integrity": "integrity"}
        )
        g.add_edge("replan", "retrieval")
        g.add_edge("integrity", END)
        return g.compile()


    def answer(self, query: str) -> dict[str, Any]:
        initial_state: OrchestratorState = {
            "query": query, "trace": [], "start_time": time.time(),
            "retries": 0, "evidence": [], "contested_warnings": [],
        }
        return self._graph.invoke(initial_state, config={"recursion_limit": 25})
