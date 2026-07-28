"""
schemas.py - Pydantic request/response models for all API endpoints.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


# ── Document Models ───────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    doc_id: str
    doc_name: str
    upload_timestamp: str
    total_pages: int
    total_chunks: int
    processing_status: str          # "pending" | "processing" | "ready" | "failed"
    file_size_kb: float
    category: Optional[str] = None
    category_confidence: Optional[float] = None


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentMetadata]


class DeleteResponse(BaseModel):
    doc_id: str
    message: str


# ── Search Models ─────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    doc_ids: Optional[List[str]] = None     # None = search all documents
    top_k: Optional[int] = 5
    search_mode: Optional[str] = "semantic" # "semantic" | "keyword" | "hybrid"


class SearchResult(BaseModel):
    doc_id: str
    doc_name: str
    page_number: int
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    search_mode: str
    results: List[SearchResult]
    total_results: int


# ── QA Models ─────────────────────────────────────────────────────────────────

class QARequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None
    session_id: Optional[str] = "default"


class Citation(BaseModel):
    doc_id: str
    doc_name: str
    page_number: int
    excerpt: str


class QAResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    confidence_score: float
    session_id: str


# ── Comparison Models ─────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    doc_ids: List[str]
    aspect: Optional[str] = "general"


class CompareResponse(BaseModel):
    doc_ids: List[str]
    doc_names: List[str]
    aspect: str
    comparison: str


# ── Summarization Models ──────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    doc_id: str
    summary_type: Optional[str] = "executive"  # executive | technical | bullet | key_takeaways


class SummarizeResponse(BaseModel):
    doc_id: str
    doc_name: str
    summary_type: str
    summary: str


# ── Classification Models ─────────────────────────────────────────────────────

class ClassifyResponse(BaseModel):
    doc_id: str
    doc_name: str
    category: str
    confidence: float
    all_scores: dict


# ── Analytics Models ──────────────────────────────────────────────────────────

class AnalyticsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_embeddings: int
    total_questions_answered: int
    most_queried_documents: List[dict]
    documents_by_category: dict
    processing_status_breakdown: dict


# ── Conversation Models ───────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    role: str       # "user" | "assistant"
    content: str
    timestamp: str


class SessionResponse(BaseModel):
    session_id: str
    history: List[ConversationTurn]


class ClearSessionResponse(BaseModel):
    session_id: str
    message: str
