from rank_bm25 import BM25Okapi

from graphitti.retrieval.base import RetrievalStrategy
from graphitti.retrieval.text_index import TextChunkStore


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class SparseBM25Strategy(RetrievalStrategy):
    name = "sparse_bm25"

    def __init__(self, store: TextChunkStore):
        self.store = store
        corpus_tokens = [_tokenize(t) for t in store.texts] or [[]]
        self.bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if not self.store.ids:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.store.ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, score in ranked[:top_k] if score > 0]
