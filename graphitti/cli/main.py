import argparse
import time

from graphitti.config import EXTRACTION_BACKEND
from graphitti.crawling.crawler import crawl_sync
from graphitti.extraction.triple_extractor import extract_triples_from_pages
from graphitti.graph.consistency import mark_contested
from graphitti.graph.loader import GraphLoader


def run(start_url: str, depth: int, pages: int, reset: bool = True):
    t0 = time.time()

    loader = GraphLoader()
    try:
        if reset:
            print("[0/4] Clearing existing graph (pass --no-reset to accumulate instead)...")
            loader.clear_all()

        print(f"[1/4] Crawling {start_url} (depth={depth}, max_pages={pages})...")
        crawled_pages = crawl_sync(start_url, max_depth=depth, max_pages=pages)
        print(f"      -> {len(crawled_pages)} pages crawled")

        print(f"[2/4] Extracting triples via '{EXTRACTION_BACKEND}' backend "
              f"({'local NLTK pipeline, no API calls' if EXTRACTION_BACKEND != 'groq' else 'LangChain LCEL chain'})...")
        triples = extract_triples_from_pages(crawled_pages)
        print(f"      -> {len(triples)} triples extracted")

        print("[3/4] Running consistency agent (conflict detection)...")
        triples, conflicts = mark_contested(triples)
        print(f"      -> {len(conflicts)} conflicting claim group(s) flagged as contested")

        print("[4/4] Loading into Neo4j...")
        batch_id = loader.load_triples(triples)
        stats = loader.stats()
    finally:
        loader.close()

    elapsed = time.time() - t0
    print(f"\nGraph spin-up time (crawl -> loaded): {elapsed:.1f}s")
    print(f"Batch ID (use for rollback): {batch_id}")
    print(f"Graph totals: {stats['entities']} entities, {stats['relations']} relations")
    if conflicts:
        print("Contested claims:")
        for c in conflicts:
            objs = sorted({cl["object"] for cl in c["claims"]})
            print(f"  - ({c['subject']}, {c['predicate']}) -> {objs}")


def rollback(batch_id: str):
    loader = GraphLoader()
    try:
        loader.delete_batch(batch_id)
        print(f"Rolled back batch {batch_id}")
    finally:
        loader.close()


def main():
    parser = argparse.ArgumentParser(description="Graphitti: crawl -> extract -> Neo4j")
    parser.add_argument("url", nargs="?", help="Seed URL to crawl")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--rollback", type=str, help="Batch ID to roll back instead of crawling")
    parser.add_argument("--no-reset", action="store_true",
                         help="Accumulate into the existing graph instead of clearing it first")
    args = parser.parse_args()

    if args.rollback:
        rollback(args.rollback)
    elif args.url:
        run(args.url, args.depth, args.pages, reset=not args.no_reset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
