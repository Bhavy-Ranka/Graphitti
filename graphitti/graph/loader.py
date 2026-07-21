import json
import logging
import uuid

from neo4j import GraphDatabase

from graphitti.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("graph_loader")

_LOAD_QUERY = """
UNWIND $triples AS t
MERGE (s:Entity {name: t.subject})
  ON CREATE SET s.type = t.subject_type
MERGE (o:Entity {name: t.object})
  ON CREATE SET o.type = t.object_type
MERGE (s)-[r:REL {
  predicate: t.predicate,
  source_url: t.source_url,
  batch_id: t.batch_id
}]->(o)
SET r.confidence = t.confidence,
    r.source_title = t.source_title,
    r.extracted_at = t.extracted_at,
    r.contested = coalesce(t.contested, false),
    r.conflicting_with = coalesce(t.conflicting_with_json, "")
"""

_DELETE_BATCH_QUERY = """
MATCH ()-[r:REL {batch_id: $batch_id}]->()
DELETE r
"""

_DELETE_ORPHANS_QUERY = """
MATCH (e:Entity)
WHERE NOT (e)--()
DELETE e
"""

_CONSTRAINT_QUERY = """
CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE
"""


class GraphLoader:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints()

    def close(self):
        self.driver.close()

    def _ensure_constraints(self):
        with self.driver.session() as session:
            session.run(_CONSTRAINT_QUERY)

    def load_triples(self, triples: list[dict], batch_size: int = 200) -> str:
        if not triples:
            log.info("No triples to load")
            return ""

        batch_id = str(uuid.uuid4())
        for t in triples:
            t["batch_id"] = batch_id
            if t.get("conflicting_with"):
                t["conflicting_with_json"] = json.dumps(t["conflicting_with"])

        with self.driver.session() as session:
            for i in range(0, len(triples), batch_size):
                chunk = triples[i:i + batch_size]
                session.execute_write(lambda tx, c=chunk: tx.run(_LOAD_QUERY, triples=c))
                log.info(f"Loaded batch chunk {i}-{i + len(chunk)}")

        log.info(f"Loaded {len(triples)} triples under batch_id={batch_id}")
        return batch_id

    def delete_batch(self, batch_id: str, drop_orphans: bool = True):
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(_DELETE_BATCH_QUERY, batch_id=batch_id))
            if drop_orphans:
                session.execute_write(lambda tx: tx.run(_DELETE_ORPHANS_QUERY))
        log.info(f"Rolled back batch_id={batch_id}")

    def clear_all(self):
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run("MATCH (n:Entity) DETACH DELETE n"))
        log.info("Cleared entire graph")

    def stats(self) -> dict:
        with self.driver.session() as session:
            nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rels = session.run("MATCH ()-[r:REL]->() RETURN count(r) AS c").single()["c"]
        return {"entities": nodes, "relations": rels}
