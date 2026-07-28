"""
search.py - Semantic, keyword, and hybrid search endpoints.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.services.rag_pipeline import rag_pipeline
from app.services.document_store import document_store
from app.core.logging import logger

router = APIRouter(prefix="/search", tags=["Search"])


def _resolve_doc_ids(requested_ids):
    """If no doc_ids specified, search all ready documents."""
    if requested_ids:
        return requested_ids
    return [
        d["doc_id"] for d in document_store.all()
        if d["processing_status"] == "ready"
    ]


@router.post("/", response_model=SearchResponse, summary="Search across documents")
async def search(request: SearchRequest):
    """
    Search documents using semantic, keyword, or hybrid strategy.

    - **semantic**: Uses vector embeddings + cosine similarity (best for conceptual queries)
    - **keyword**: Uses word frequency matching (best for exact term lookup)
    - **hybrid**: Combines both strategies (0.7 semantic + 0.3 keyword weight)
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    doc_ids = _resolve_doc_ids(request.doc_ids)
    if not doc_ids:
        raise HTTPException(status_code=404, detail="No processed documents available to search.")

    top_k = request.top_k or 5
    mode  = request.search_mode or "semantic"

    if mode == "semantic":
        raw_results = rag_pipeline.semantic_search(request.query, doc_ids, top_k)
    elif mode == "keyword":
        raw_results = rag_pipeline.keyword_search(request.query, doc_ids, top_k)
    elif mode == "hybrid":
        raw_results = rag_pipeline.hybrid_search(request.query, doc_ids, top_k)
    else:
        raise HTTPException(status_code=400, detail="search_mode must be 'semantic', 'keyword', or 'hybrid'.")

    results = []
    for r in raw_results:
        doc = document_store.get(r["doc_id"])
        doc_name = doc["doc_name"] if doc else r["doc_id"]
        results.append(SearchResult(
            doc_id=r["doc_id"],
            doc_name=doc_name,
            page_number=r["page_number"],
            chunk_index=r["chunk_index"],
            text=r["text"],
            score=round(r["score"], 4),
        ))

    logger.info(f"Search '{request.query[:50]}' → {len(results)} results via {mode}")
    return SearchResponse(
        query=request.query,
        search_mode=mode,
        results=results,
        total_results=len(results),
    )
