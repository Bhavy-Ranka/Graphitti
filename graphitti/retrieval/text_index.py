from __future__ import annotations

from graphitti.extraction.chunking import semantic_chunk


class TextChunkStore:
    def __init__(self):
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.meta: dict[str, dict] = {}

    def add_pages(self, pages: list[dict]):
        for page in pages:
            text = page.get("text") or page.get("page_text") or ""
            chunks = semantic_chunk(text)
            for i, chunk in enumerate(chunks):
                cid = f"{page['url']}#chunk{i}"
                self.ids.append(cid)
                self.texts.append(chunk)
                self.meta[cid] = {"url": page["url"], "title": page.get("title", ""), "chunk_index": i}

    def clear(self):
        self.ids, self.texts, self.meta = [], [], {}

    def __len__(self):
        return len(self.ids)
