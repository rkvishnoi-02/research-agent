"""LangChain-compatible retrieval over collected evidence."""

from __future__ import annotations

from collections import Counter

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from .models import RawQuote


class LexicalEvidenceRetriever(BaseRetriever):
    """A lightweight retriever that stays LangChain-compatible without embeddings."""

    documents: list[Document] = Field(default_factory=list)
    k: int = 4

    def _get_relevant_documents(self, query: str) -> list[Document]:
        query_terms = Counter(query.lower().split())
        scored: list[tuple[int, Document]] = []
        for document in self.documents:
            text_terms = Counter(document.page_content.lower().split())
            score = sum((query_terms & text_terms).values())
            if score:
                scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[: self.k]]

    @classmethod
    def from_quotes(cls, quotes: list[RawQuote], k: int = 4) -> "LexicalEvidenceRetriever":
        documents = [
            Document(
                page_content=quote.text,
                metadata={
                    "quote_id": quote.quote_id,
                    "category": quote.category,
                    "source_type": quote.source_type,
                    "source_url": quote.source_url,
                    "context_snippet": quote.context_snippet,
                },
            )
            for quote in quotes
        ]
        return cls(documents=documents, k=k)
