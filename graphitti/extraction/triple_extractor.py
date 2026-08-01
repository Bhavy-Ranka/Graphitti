import logging
from datetime import datetime, timezone

from graphitti.extraction.chunking import semantic_chunk
from graphitti.extraction.direction import normalize_direction
from graphitti.extraction.entity_resolution import resolve_entity_aliases
from graphitti.extraction.nlp_pipeline import extract_triples_from_chunk, normalize_entity_types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("triple_extractor")

def _extract_chunk(chunk_text: str) -> list[dict]:
    try:
        return extract_triples_from_chunk(chunk_text)
    except Exception as e:
        log.warning(f"NLTK extraction failed for chunk, skipping: {e}")
        return []

def extract_triples_from_page(page: dict) -> list[dict]:
    chunks = semantic_chunk(page["text"])
    all_triples = []

    for i, chunk in enumerate(chunks):
        for t in _extract_chunk(chunk):
            if not t.get("subject") or not t.get("object"):
                continue
            triple = {
                "subject": t["subject"].strip(),
                "subject_type": t.get("subject_type", "Other"),
                "predicate": t["predicate"].strip().lower().replace(" ", "_"),
                "object": t["object"].strip(),
                "object_type": t.get("object_type", "Other"),
                "confidence": float(t.get("confidence", 0.5)),
                "source_url": page["url"],
                "source_title": page.get("title", ""),
                "chunk_index": i,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }
            all_triples.append(normalize_direction(triple))

    all_triples = normalize_entity_types(all_triples)

    all_triples = resolve_entity_aliases(all_triples)

    log.info(f"Extracted {len(all_triples)} triples from {page['url']} ({len(chunks)} chunks)")
    return all_triples


def extract_triples_from_pages(pages: list[dict]) -> list[dict]:
    all_triples = []
    for page in pages:
        all_triples.extend(extract_triples_from_page(page))
    return resolve_entity_aliases(all_triples)
