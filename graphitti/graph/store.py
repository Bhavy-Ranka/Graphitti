from __future__ import annotations
import difflib
from abc import ABC, abstractmethod
import networkx as nx
from graphitti.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USE

class GraphStore(ABC):
    @abstractmethod
    def match_entities(self, query: str, top_k: int = 3) -> list[str]:
        pass
      
    @abstractmethod
    def neighbors(self, entity_name: str) -> list[dict]:
        pass

    @abstractmethod
    def bfs(self, seed_entities: list[str], max_hops: int = 2) -> list[dict]:
        pass

    @abstractmethod
    def contested_edges(self) -> list[dict]:
        pass


def _fuzzy_match(name_pool: list[str], query: str, top_k: int) -> list[str]:
    q = query.lower()
    substring_hits = [n for n in name_pool if n.lower() in q or q in n.lower()]
    if len(substring_hits) >= top_k:
        return substring_hits[:top_k]
    candidates = set(substring_hits)
    tokens = [q] + q.split()
    for tok in tokens:
        candidates.update(difflib.get_close_matches(tok, name_pool, n=top_k, cutoff=0.6))
        if len(candidates) >= top_k:
            break
    return list(candidates)[:top_k]

class InMemoryGraphStore(GraphStore):
    def __init__(self):
        self.g = nx.MultiDiGraph()
    def clear(self):
        self.g = nx.MultiDiGraph()
    def load_triples(self, triples: list[dict], batch_id: str | None = None):
        for t in triples:
            s, o = t["subject"], t["object"]
            self.g.add_node(s, type=t.get("subject_type", "Other"))
            self.g.add_node(o, type=t.get("object_type", "Other"))
            self.g.add_edge(
                s, o,
                predicate=t["predicate"],
                confidence=t.get("confidence", 0.5),
                source_url=t.get("source_url", ""),
                source_title=t.get("source_title", ""),
                batch_id=batch_id or t.get("batch_id", ""),
                contested=t.get("contested", False),
            )

    def match_entities(self, query: str, top_k: int = 3) -> list[str]:
        return _fuzzy_match(list(self.g.nodes), query, top_k)

    def neighbors(self, entity_name: str) -> list[dict]:
        out = []
        if entity_name not in self.g:
            return out
        for _, nbr, data in self.g.out_edges(entity_name, data=True):
            out.append({"neighbor": nbr, "predicate": data["predicate"], "direction": "out",
                        "confidence": data["confidence"], "source_url": data["source_url"]})
        for nbr, _, data in self.g.in_edges(entity_name, data=True):
            out.append({"neighbor": nbr, "predicate": data["predicate"], "direction": "in",
                        "confidence": data["confidence"], "source_url": data["source_url"]})
        return out

    def bfs(self, seed_entities: list[str], max_hops: int = 2) -> list[dict]:
        visited: dict[str, int] = {}
        frontier = [e for e in seed_entities if e in self.g]
        for e in frontier:
            visited[e] = 0
        hop = 0
        while frontier and hop < max_hops:
            hop += 1
            next_frontier = []
            for node in frontier:
                for _, nbr in list(self.g.out_edges(node)) + [(b, a) for a, b in self.g.in_edges(node)]:
                    if nbr not in visited:
                        visited[nbr] = hop
                        next_frontier.append(nbr)
            frontier = next_frontier
        return [{"entity": e, "hops": h} for e, h in sorted(visited.items(), key=lambda x: x[1])]

    def contested_edges(self) -> list[dict]:
        out = []
        for u, v, data in self.g.edges(data=True):
            if data.get("contested"):
                out.append({"subject": u, "object": v, **data})
        return out

    def all_entities(self) -> list[str]:
        return list(self.g.nodes)

    def export_for_viz(self) -> dict:
        nodes = [{"id": n, "type": d.get("type", "Other")} for n, d in self.g.nodes(data=True)]
        edges = [{"source": u, "target": v, "predicate": d["predicate"], "contested": d.get("contested", False)}
                  for u, v, d in self.g.edges(data=True)]
        return {"nodes": nodes, "edges": edges}

class Neo4jGraphStore(GraphStore):
    def __init__(self, driver=None):
        if driver is None:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.driver = driver

    def close(self):
        self.driver.close()

    def _entity_names(self) -> list[str]:
        with self.driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN e.name AS name")
            return [r["name"] for r in result]

    def match_entities(self, query: str, top_k: int = 3) -> list[str]:
        return _fuzzy_match(self._entity_names(), query, top_k)

    def neighbors(self, entity_name: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $name})-[r:REL]-(n:Entity)
                RETURN n.name AS neighbor, r.predicate AS predicate,
                       CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END AS direction,
                       r.confidence AS confidence, r.source_url AS source_url
                """,
                name=entity_name,
            )
            return [dict(r) for r in result]

    def bfs(self, seed_entities: list[str], max_hops: int = 2) -> list[dict]:
        max_hops = int(max_hops)
        with self.driver.session() as session:
            result = session.run(
                f"""
                UNWIND $seeds AS seed
                MATCH (s:Entity {{name: seed}})
                CALL {{
                  WITH s
                  MATCH p = (s)-[:REL*1..{max_hops}]-(n:Entity)
                  RETURN n.name AS entity, min(length(p)) AS hops
                }}
                RETURN DISTINCT entity, hops ORDER BY hops
                """,
                seeds=seed_entities,
            )
            return [dict(r) for r in result]

    def contested_edges(self) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Entity)-[r:REL {contested: true}]->(o:Entity)
                RETURN s.name AS subject, o.name AS object, r.predicate AS predicate,
                       r.confidence AS confidence, r.source_url AS source_url
                """
            )
            return [dict(r) for r in result]


def get_graph_store() -> GraphStore:
    return Neo4jGraphStore()
