from graphitti.graph.store import GraphStore
from graphitti.retrieval.base import RetrievalStrategy


class GraphTraversalStrategy(RetrievalStrategy):
    name = "graph_traversal"

    def __init__(self, store: GraphStore, max_hops: int = 2):
        self.store = store
        self.max_hops = max_hops

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        seeds = self.store.match_entities(query, top_k=3)
        if not seeds:
            return []
        visited = self.store.bfs(seeds, max_hops=self.max_hops)
        return [v["entity"] for v in visited[:top_k]]


class EntityCentricStrategy(RetrievalStrategy):
    name = "entity_centric"

    def __init__(self, store: GraphStore):
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        matches = self.store.match_entities(query, top_k=1)
        if not matches:
            return []
        entity = matches[0]
        neighbors = self.store.neighbors(entity)
        neighbors.sort(key=lambda n: n.get("confidence", 0), reverse=True)
        ids = [entity] + [n["neighbor"] for n in neighbors]
        seen = set()
        ordered = [x for x in ids if not (x in seen or seen.add(x))]
        return ordered[:top_k]
