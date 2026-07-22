from __future__ import annotations

import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from graphitti.retrieval.base import RetrievalStrategy
from graphitti.retrieval.text_index import TextChunkStore

log = logging.getLogger("dense_retrieval")


class DenseVectorStrategy(RetrievalStrategy):
    name = "dense_vector"

    def __init__(self, store: TextChunkStore, try_sentence_transformers: bool = True):
        self.store = store
        self.backend = "tfidf"
        self._model = None
        self._doc_vectors = None
        self._vectorizer = None

        if try_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self._doc_vectors = self._model.encode(store.texts or [""])
                self.backend = "sentence-transformers"
                return
            except Exception as e:
                log.info(f"sentence-transformers unavailable ({e}); falling back to TF-IDF")

        self._vectorizer = TfidfVectorizer(stop_words="english")
        corpus = store.texts or [""]
        self._doc_vectors = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        if not self.store.ids:
            return []

        if self.backend == "sentence-transformers":
            q_vec = self._model.encode([query])
            sims = cosine_similarity(q_vec, self._doc_vectors)[0]
        else:
            q_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self._doc_vectors)[0]

        ranked = sorted(zip(self.store.ids, sims), key=lambda x: x[1], reverse=True)
        return [cid for cid, score in ranked[:top_k] if score > 0]
