"""Backend service tools.

Keep this package import side-effect free. Several modules import
``backend.services.*`` during FastAPI startup, and eager RAG initialization can
load Chroma/embedding clients before API keys or runtime state are ready.
"""

from __future__ import annotations

from langchain_core.tools import tool


_rag_service = None


def get_rag_service():
    global _rag_service
    if _rag_service is None:
        from backend.services.rag.rag_service import RagSummarizeService

        _rag_service = RagSummarizeService()
    return _rag_service


@tool(description="RAG query for Goose Goose Duck rules and terms.")
async def rag_query(query: str) -> str:
    return await get_rag_service().arag_summarize(query)
