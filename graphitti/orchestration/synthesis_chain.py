from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from graphitti.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE

SYNTHESIS_SYSTEM = """You answer a user's question using ONLY the numbered evidence items given.
Rules:
- Write 2-5 sentences of plain, direct prose that actually answers the question.
- After each claim, cite the evidence item(s) it came from like [1] or [1,3].
- Do not invent anything not supported by the evidence.
- If the evidence doesn't actually answer the question, say so plainly instead of padding."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYNTHESIS_SYSTEM),
    ("human", "Question: {query}\n\nEvidence:\n{evidence_block}"),
])

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=GROQ_TEMPERATURE)
        _chain = _prompt | llm | StrOutputParser()
    return _chain


def _format_evidence_block(evidence: list[dict]) -> str:
    lines = []
    for i, e in enumerate(evidence, start=1):
        src = e.get("source_url") or "graph"
        lines.append(f"[{i}] {e['text']} (source: {src})")
    return "\n".join(lines)


def _fallback_answer(evidence: list[dict]) -> str:
    lines = [f"- {e['text']} [source: {e.get('source_url') or 'graph'}]" for e in evidence]
    return "Based on the retrieved evidence:\n" + "\n".join(lines)


def synthesize_answer(query: str, evidence: list[dict]) -> str:
    if not evidence:
        return "No relevant information found in the graph for this query."
    if not GROQ_API_KEY:
        return _fallback_answer(evidence)
    try:
        return _get_chain().invoke({
            "query": query,
            "evidence_block": _format_evidence_block(evidence),
        }).strip()
    except Exception:
        return _fallback_answer(evidence)
