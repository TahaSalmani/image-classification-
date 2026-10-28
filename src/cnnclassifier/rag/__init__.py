"""Retrieval-augmented grounding (RAG) for the pneumonia knowledge base.

Stage 1 (``chunker``) turns the sourced markdown files under ``knowledge/``
into short passages that keep their ``SOURCE:`` URLs, and stage 2
(``retriever``) ranks those passages against a question with TF-IDF.
"""

from .chunker import Chunk, load_knowledge_base
from .retriever import Hit, KnowledgeRetriever, get_retriever

__all__ = [
    "Chunk",
    "Hit",
    "KnowledgeRetriever",
    "get_retriever",
    "load_knowledge_base",
]
