"""
analytics.py - Analytics and knowledge base statistics endpoints.
"""

from fastapi import APIRouter
from app.models.schemas import AnalyticsResponse
from app.services.document_store import document_store
from app.services.analytics import analytics_tracker

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/", response_model=AnalyticsResponse, summary="Get knowledge base analytics")
async def get_analytics():
    """
    Returns usage statistics and knowledge base metrics.
    """
    docs  = document_store.all()
    stats = analytics_tracker.get_stats()

    total_chunks     = sum(d.get("total_chunks", 0) for d in docs)
    total_embeddings = total_chunks   # 1 embedding per chunk

    # Category distribution
    docs_by_category: dict = {}
    for d in docs:
        cat = d.get("category") or "Unclassified"
        docs_by_category[cat] = docs_by_category.get(cat, 0) + 1

    # Processing status breakdown
    status_counts: dict = {}
    for d in docs:
        s = d.get("processing_status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    # Enrich most queried with doc names
    most_queried_raw = analytics_tracker.get_most_queried(top_n=5)
    most_queried = []
    for item in most_queried_raw:
        doc = document_store.get(item["doc_id"])
        most_queried.append({
            "doc_id": item["doc_id"],
            "doc_name": doc["doc_name"] if doc else item["doc_id"],
            "query_count": item["query_count"],
        })

    return AnalyticsResponse(
        total_documents=len(docs),
        total_chunks=total_chunks,
        total_embeddings=total_embeddings,
        total_questions_answered=stats.get("total_questions_answered", 0),
        most_queried_documents=most_queried,
        documents_by_category=docs_by_category,
        processing_status_breakdown=status_counts,
    )
