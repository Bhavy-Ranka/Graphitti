import os

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


MAX_DEPTH = int(os.getenv("MAX_DEPTH", "2"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "20"))
REQUEST_TIMEOUT_MS = int(os.getenv("REQUEST_TIMEOUT_MS", "15000"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "180"))

<<<<<<< HEAD
=======
# Triple extraction backend: "nltk" (default) runs entirely locally, no API
# calls, no rate limits, no per-token latency -- see
# graphitti/extraction/nlp_pipeline.py. Set to "groq" to use the original
# LangChain + ChatGroq LLM extractor instead.
>>>>>>> 798fdaf (final project)
EXTRACTION_BACKEND = os.getenv("EXTRACTION_BACKEND", "nltk").lower()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
GRAPH_MAX_HOPS = int(os.getenv("GRAPH_MAX_HOPS", "2"))

ORCHESTRATOR_LATENCY_CEILING_S = float(os.getenv("ORCHESTRATOR_LATENCY_CEILING_S", "20"))
ORCHESTRATOR_MAX_RETRIES = int(os.getenv("ORCHESTRATOR_MAX_RETRIES", "2"))

ROUTING_LOG_PATH = os.getenv("ROUTING_LOG_PATH", "routing_log.jsonl")
