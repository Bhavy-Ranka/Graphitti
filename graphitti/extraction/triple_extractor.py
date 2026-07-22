import logging
from datetime import datetime, timezone

from graphitti.config import EXTRACTION_BACKEND
from graphitti.extraction.chunking import semantic_chunk
from graphitti.extraction.direction import normalize_direction
from graphitti.extraction.entity_resolution import resolve_entity_aliases

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("triple_extractor")

<<<<<<< HEAD
=======

# ---------------------------------------------------------------------------
# Backend: local NLTK pipeline (default) -- no API calls, no rate limits.
# ---------------------------------------------------------------------------

>>>>>>> 798fdaf (final project)
def _extract_chunk_nltk(chunk_text: str) -> list[dict]:
    from graphitti.extraction.nlp_pipeline import extract_triples_from_chunk
    try:
        return extract_triples_from_chunk(chunk_text)
    except Exception as e:
        log.warning(f"NLTK extraction failed for chunk, skipping: {e}")
        return []

<<<<<<< HEAD
=======

# ---------------------------------------------------------------------------
# Backend: Groq LLM via LangChain (opt-in, EXTRACTION_BACKEND=groq)
# ---------------------------------------------------------------------------

>>>>>>> 798fdaf (final project)
_groq_chain = None

_GROQ_SYSTEM_PROMPT = """You are an information extraction engine. Given a passage of web text, \
extract factual (subject, predicate, object) triples.

Rules:
- Only extract triples that are explicitly stated or directly implied by the text.
- predicate should be a short, normalized verb phrase (e.g. "founded", "based_in", "released").
- Direction matters: the subject should be the entity the predicate is fundamentally about, not
  whichever entity happens to appear first in the sentence. Write "Person born_in Location", not
  "Location born_in Person"; "Person plays_for Team", not "Team plays_for Person"; "Person
  born_on Date", not "Date born_on Person".
- confidence reflects how explicit/certain the claim is in the text.
- If the passage contains no extractable facts, return an empty triple list."""


def _build_groq_chain():
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq

    from graphitti.config import GROQ_API_KEY, GROQ_MODEL
    from graphitti.extraction.schemas import TripleList

    prompt = ChatPromptTemplate.from_messages([
        ("system", _GROQ_SYSTEM_PROMPT),
        ("human", "{chunk_text}"),
    ])
    llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.1)
    return prompt | llm.with_structured_output(TripleList)


def _get_groq_chain():
    global _groq_chain
    if _groq_chain is None:
        from graphitti.config import GROQ_API_KEY
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. EXTRACTION_BACKEND=groq requires a "
                "Groq API key (see .env.example), or unset EXTRACTION_BACKEND "
                "to use the local NLTK pipeline instead."
            )
        _groq_chain = _build_groq_chain()
    return _groq_chain


def _extract_chunk_groq(chunk_text: str) -> list[dict]:
    try:
        result = _get_groq_chain().invoke({"chunk_text": chunk_text})
        return [t.model_dump() for t in result.triples]
    except Exception as e:
        log.warning(f"Groq extraction call failed for chunk, skipping: {e}")
        return []


def _extract_chunk(chunk_text: str) -> list[dict]:
    if EXTRACTION_BACKEND == "groq":
        return _extract_chunk_groq(chunk_text)
    return _extract_chunk_nltk(chunk_text)

<<<<<<< HEAD
=======

# ---------------------------------------------------------------------------
# Public API (backend-agnostic)
# ---------------------------------------------------------------------------

>>>>>>> 798fdaf (final project)
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

    if EXTRACTION_BACKEND != "groq":
        from graphitti.extraction.nlp_pipeline import normalize_entity_types
        all_triples = normalize_entity_types(all_triples)

    all_triples = resolve_entity_aliases(all_triples)

    log.info(f"Extracted {len(all_triples)} triples from {page['url']} ({len(chunks)} chunks)")
    return all_triples


def extract_triples_from_pages(pages: list[dict]) -> list[dict]:
    all_triples = []
    for page in pages:
        all_triples.extend(extract_triples_from_page(page))
    return resolve_entity_aliases(all_triples)
