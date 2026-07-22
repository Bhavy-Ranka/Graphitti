import json
import random
import statistics

from graphitti.routing.query_router import QueryRouter


def precision_recall(retrieved: list[str], relevant: list[str]):
    if not retrieved:
        return 0.0, 0.0
    retrieved_set, relevant_set = set(retrieved), set(relevant)
    hits = len(retrieved_set & relevant_set)
    precision = hits / len(retrieved_set)
    recall = hits / len(relevant_set) if relevant_set else 0.0
    return precision, recall


def load_test_set(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_synthetic_test_set(text_store, graph_store, n: int = 15, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    entities = graph_store.all_entities() if hasattr(graph_store, "all_entities") else []
    entities = rng.sample(entities, min(n, len(entities))) if entities else []

    test_set = []
    for ent in entities:
        relevant_chunks = [cid for cid, text in zip(text_store.ids, text_store.texts) if ent.lower() in text.lower()]
        neighbors = [n_["neighbor"] for n_ in graph_store.neighbors(ent)]
        test_set.append({
            "query": f"What is known about {ent}?",
            "relevant_ids": relevant_chunks + [ent] + neighbors,
        })
    return test_set


def run_fixed_strategy(strategy, test_set, top_k=5):
    results = []
    for item in test_set:
        retrieved, latency_ms = strategy.timed_retrieve(item["query"], top_k)
        precision, recall = precision_recall(retrieved, item["relevant_ids"])
        results.append({"query": item["query"], "precision": precision, "recall": recall, "latency_ms": latency_ms})
    return results


def run_routed(router: QueryRouter, strategies: dict, test_set, top_k=5):
    results = []
    for item in test_set:
        decision = router.route(item["query"])
        strategy = strategies[decision["chosen_strategy"]]
        retrieved, latency_ms = strategy.timed_retrieve(decision["rephrased_query"], top_k)
        precision, recall = precision_recall(retrieved, item["relevant_ids"])
        results.append({
            "query": item["query"],
            "chosen_strategy": decision["chosen_strategy"],
            "intent": decision["intent"],
            "precision": precision,
            "recall": recall,
            "latency_ms": latency_ms + decision["routing_latency_ms"],
        })
    return results


def summarize(results, label):
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]
    latencies = [r["latency_ms"] for r in results]
    return {
        "strategy": label,
        "avg_precision": round(statistics.mean(precisions), 4) if precisions else 0.0,
        "avg_recall": round(statistics.mean(recalls), 4) if recalls else 0.0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "n_queries": len(results),
    }


def run_ablation(test_set: list[dict], strategies: dict, router: QueryRouter, top_k=5):
    report = []
    for name, strategy in strategies.items():
        if name in ("graph_1hop",):
            continue
        results = run_fixed_strategy(strategy, test_set, top_k)
        report.append(summarize(results, name))

    routed_results = run_routed(router, strategies, test_set, top_k)
    report.append(summarize(routed_results, "routed (this system)"))
    return report


def main():
    import sys
    from pathlib import Path

    from graphitti.graph.consistency import mark_contested
    from graphitti.graph.store import InMemoryGraphStore
    from graphitti.extraction.triple_extractor import extract_triples_from_pages
    from graphitti.retrieval.base import build_strategy_registry
    from graphitti.retrieval.text_index import TextChunkStore

    dataset_path = Path(__file__).resolve().parents[2] / "crawled_dataset.json"
    pages = json.loads(dataset_path.read_text())

    text_store = TextChunkStore()
    text_store.add_pages(pages)

    graph = InMemoryGraphStore()
    triples = extract_triples_from_pages(pages)
    triples, _ = mark_contested(triples)
    graph.load_triples(triples, batch_id="ablation-run")

    strategies = build_strategy_registry(text_store, graph)
    router = QueryRouter()

    test_set_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_set = load_test_set(test_set_path) if test_set_path else build_synthetic_test_set(text_store, graph)

    report = run_ablation(test_set, strategies, router)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
