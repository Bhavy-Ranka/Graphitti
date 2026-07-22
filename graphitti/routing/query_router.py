import json
import os
import re
import time
import uuid
from datetime import datetime, timezone

from graphitti.config import GROQ_API_KEY, ROUTING_LOG_PATH
from graphitti.routing.chains import get_classify_chain, get_decompose_chain, get_rephrase_chain, _payload

INTENTS = [
    "single_fact_lookup",
    "entity_centric",
    "multi_hop_relational",
    "comparison",
    "temporal_versioned",
    "broad_exploratory",
    "keyword_exact",
]

STRATEGY_MAP = {
    "single_fact_lookup": "graph_1hop",
    "entity_centric": "entity_centric",
    "multi_hop_relational": "graph_traversal",
    "comparison": "hybrid_fusion",
    "temporal_versioned": "graph_traversal",
    "broad_exploratory": "dense_vector",
    "keyword_exact": "sparse_bm25",
}

_COMPARISON_RE = re.compile(r"\b(vs\.?|versus|compare|difference between)\b", re.I)
_TEMPORAL_RE = re.compile(r"\b(v\d+|version|changelog|since|as of|latest|history of)\b", re.I)
_MULTIHOP_RE = re.compile(r"\bwho (founded|acquired|created) the .* that\b|.* that (acquired|founded|owns)", re.I)
_SINGLE_FACT_RE = re.compile(r"^(who|what|when|where) (is|was|are)\b", re.I)


def _heuristic_classify(query: str) -> dict:
    if _COMPARISON_RE.search(query):
        return {"intent": "comparison", "confidence": 0.6, "reasoning": "heuristic: comparison keyword"}
    if _MULTIHOP_RE.search(query):
        return {"intent": "multi_hop_relational", "confidence": 0.6, "reasoning": "heuristic: nested clause"}
    if _TEMPORAL_RE.search(query):
        return {"intent": "temporal_versioned", "confidence": 0.55, "reasoning": "heuristic: version/time keyword"}
    if _SINGLE_FACT_RE.search(query):
        return {"intent": "single_fact_lookup", "confidence": 0.55, "reasoning": "heuristic: wh-question pattern"}
    if len(query.split()) <= 3:
        return {"intent": "keyword_exact", "confidence": 0.5, "reasoning": "heuristic: short query"}
    return {"intent": "broad_exploratory", "confidence": 0.4, "reasoning": "heuristic: no pattern matched"}


def classify_intent(query: str, conversation_context: str = "") -> dict:
    if not GROQ_API_KEY:
        return _heuristic_classify(query)
    try:
        result = get_classify_chain().invoke({"payload": _payload(query, conversation_context)})
        data = result.model_dump()
        if data["intent"] not in INTENTS:
            data["intent"], data["confidence"] = "broad_exploratory", 0.3
        return data
    except Exception as e:
        return {"intent": "broad_exploratory", "confidence": 0.0, "reasoning": f"classification_failed: {e}"}


def rephrase_query(query: str, conversation_context: str = "") -> str:
    if not GROQ_API_KEY:
        return query
    try:
        return get_rephrase_chain().invoke({"payload": _payload(query, conversation_context)}).strip()
    except Exception:
        return query


def decompose_multihop(query: str) -> list[dict]:
    if not GROQ_API_KEY:
        return [{"step": 1, "sub_query": query, "depends_on": None}]
    try:
        result = get_decompose_chain().invoke({"payload": query})
        steps = [s.model_dump() for s in result.steps]
        if steps:
            return steps
    except Exception:
        pass
    return [{"step": 1, "sub_query": query, "depends_on": None}]


class RoutingLogger:
    def __init__(self, path=ROUTING_LOG_PATH):
        self.path = path

    def log(self, decision: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(decision) + "\n")

    def read_all(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]


class QueryRouter:
    def __init__(self, confidence_floor: float = 0.55, logger: RoutingLogger = None):
        self.confidence_floor = confidence_floor
        self.logger = logger or RoutingLogger()

    def route(self, query: str, conversation_context: str = "") -> dict:
        t0 = time.time()
        request_id = str(uuid.uuid4())

        intent_result = classify_intent(query, conversation_context)
        intent = intent_result["intent"]
        confidence = intent_result.get("confidence", 0.0)
        reasoning = intent_result.get("reasoning", "")

        rephrased = rephrase_query(query, conversation_context)

        sub_queries = None
        if intent == "multi_hop_relational":
            sub_queries = decompose_multihop(rephrased)

        strategy = STRATEGY_MAP.get(intent, "hybrid_fusion")
        fallback_used = False
        if confidence < self.confidence_floor:
            strategy = "hybrid_fusion"
            fallback_used = True

        decision = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_query": query,
            "rephrased_query": rephrased,
            "intent": intent,
            "intent_confidence": confidence,
            "intent_reasoning": reasoning,
            "chosen_strategy": strategy,
            "fallback_used": fallback_used,
            "sub_queries": sub_queries,
            "routing_latency_ms": round((time.time() - t0) * 1000, 2),
        }

        self.logger.log(decision)
        return decision
