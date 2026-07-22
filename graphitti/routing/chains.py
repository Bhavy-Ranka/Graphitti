from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from graphitti.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE
from graphitti.routing.schemas import DecomposedQuery, IntentClassification

CLASSIFY_SYSTEM = """Classify the user's query into exactly one intent from this list:
single_fact_lookup, entity_centric, multi_hop_relational, comparison, temporal_versioned, broad_exploratory, keyword_exact

single_fact_lookup: asks for one attribute of one entity ("who is the CEO of X")
entity_centric: wants everything about one entity ("tell me about X")
multi_hop_relational: needs traversing 2+ relations to answer ("who founded the company that acquired X")
comparison: compares two or more entities ("difference between X and Y")
temporal_versioned: asks about a specific version/date/change over time ("what changed in v2 vs v1")
broad_exploratory: vague, open-ended, no clear single entity ("what's interesting about X's ecosystem")
keyword_exact: looking for an exact term/phrase/code identifier match"""

REPHRASE_SYSTEM = """Rewrite the user's query into a clean, unambiguous search query for a
knowledge graph retrieval system. Resolve pronouns if context is given, strip filler words,
keep named entities exact. Return ONLY the rewritten query text, nothing else."""

DECOMPOSE_SYSTEM = """The query requires multiple hops across a knowledge graph to answer.
Break it into an ordered list of atomic sub-queries, where each later sub-query may depend on
the answer of an earlier one (depends_on = step number it depends on, or null)."""

_HUMAN_WITH_CONTEXT = "{payload}"


def _llm():
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=GROQ_TEMPERATURE)


def _payload(query: str, conversation_context: str = "") -> str:
    return query if not conversation_context else f"Context: {conversation_context}\nQuery: {query}"


_classify_chain = None
_rephrase_chain = None
_decompose_chain = None


def get_classify_chain():
    global _classify_chain
    if _classify_chain is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", CLASSIFY_SYSTEM), ("human", _HUMAN_WITH_CONTEXT),
        ])
        _classify_chain = prompt | _llm().with_structured_output(IntentClassification)
    return _classify_chain


def get_rephrase_chain():
    global _rephrase_chain
    if _rephrase_chain is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", REPHRASE_SYSTEM), ("human", _HUMAN_WITH_CONTEXT),
        ])
        _rephrase_chain = prompt | _llm() | StrOutputParser()
    return _rephrase_chain


def get_decompose_chain():
    global _decompose_chain
    if _decompose_chain is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", DECOMPOSE_SYSTEM), ("human", _HUMAN_WITH_CONTEXT),
        ])
        _decompose_chain = prompt | _llm().with_structured_output(DecomposedQuery)
    return _decompose_chain
