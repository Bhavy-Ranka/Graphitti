import os
from pathlib import Path
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env")
load_dotenv()

def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")

MAX_DEPTH = int(os.getenv("MAX_DEPTH", "2"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "20"))
REQUEST_TIMEOUT_MS = int(os.getenv("REQUEST_TIMEOUT_MS", "15000"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "180"))

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
GRAPH_MAX_HOPS = int(os.getenv("GRAPH_MAX_HOPS", "2"))

ORCHESTRATOR_LATENCY_CEILING_S = float(os.getenv("ORCHESTRATOR_LATENCY_CEILING_S", "20"))
ORCHESTRATOR_MAX_RETRIES = int(os.getenv("ORCHESTRATOR_MAX_RETRIES", "2"))
