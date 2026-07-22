import time
from abc import ABC, abstractmethod


class RetrievalStrategy(ABC):
    name = "base"

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        raise NotImplementedError

    def timed_retrieve(self, query: str, top_k: int = 5):
        t0 = time.time()
        results = self.retrieve(query, top_k)
        latency_ms = (time.time() - t0) * 1000
        return results, latency_ms


class HybridFusionStrategy(RetrievalStrategy):
    name = "hybrid_fusion"

    def __init__(self, strategies: list[RetrievalStrategy], k_rrf: int = 60):
        self.strategies = strategies
        self.k_rrf = k_rrf

    def retrieve(self, query, top_k=5):
        scores = {}
        for strategy in self.strategies:
            ranked_ids, _ = strategy.timed_retrieve(query, top_k)
            for rank, doc_id in enumerate(ranked_ids):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.k_rrf + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:top_k]]


def build_strategy_registry(text_store, graph_store, max_hops: int = 2) -> dict:
    from graphitti.retrieval.bm25 import SparseBM25Strategy
    from graphitti.retrieval.dense import DenseVectorStrategy
    from graphitti.retrieval.graph_retrieval import EntityCentricStrategy, GraphTraversalStrategy

    dense = DenseVectorStrategy(text_store)
    bm25 = SparseBM25Strategy(text_store)
    graph_trav = GraphTraversalStrategy(graph_store, max_hops=max_hops)
    graph_1hop = GraphTraversalStrategy(graph_store, max_hops=1)
    graph_1hop.name = "graph_1hop"
    entity = EntityCentricStrategy(graph_store)

    strategies = {
        "dense_vector": dense,
        "sparse_bm25": bm25,
        "graph_traversal": graph_trav,
        "graph_1hop": graph_1hop,
        "entity_centric": entity,
    }
    strategies["hybrid_fusion"] = HybridFusionStrategy(list(strategies.values()))
    return strategies


STRATEGY_REGISTRY = {
    "dense_vector": "graphitti.retrieval.dense.DenseVectorStrategy",
    "sparse_bm25": "graphitti.retrieval.bm25.SparseBM25Strategy",
    "graph_traversal": "graphitti.retrieval.graph_retrieval.GraphTraversalStrategy",
    "graph_1hop": "graphitti.retrieval.graph_retrieval.GraphTraversalStrategy",
    "entity_centric": "graphitti.retrieval.graph_retrieval.EntityCentricStrategy",
    "hybrid_fusion": "graphitti.retrieval.base.HybridFusionStrategy",
}
