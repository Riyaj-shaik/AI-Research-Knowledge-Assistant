"""
assistant.py - AI assistant endpoints: QA, comparison, summarization, conversation.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    QARequest, QAResponse, Citation,
    CompareRequest, CompareResponse,
    SummarizeRequest, SummarizeResponse,
    SessionResponse, ClearSessionResponse,
)
from app.services.rag_pipeline import rag_pipeline
from app.services.ai_service import ai_service
from app.services.document_store import document_store
from app.services.conversation_memory import conversation_memory
from app.services.analytics import analytics_tracker
from app.core.logging import logger

router = APIRouter(prefix="/assistant", tags=["AI Assistant"])


def _get_ready_doc_ids(requested_ids=None):
    if requested_ids:
        for doc_id in requested_ids:
            doc = document_store.get(doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
            if doc["processing_status"] != "ready":
                raise HTTPException(status_code=400, detail=f"Document {doc_id} is not ready yet.")
        return requested_ids
    return [d["doc_id"] for d in document_store.all() if d["processing_status"] == "ready"]


# ── Question Answering ────────────────────────────────────────────────────────

@router.post("/ask", response_model=QAResponse, summary="Ask a question")
async def ask_question(request: QARequest):
    """
    Ask a question and get a grounded answer with citations.
    Supports multi-turn conversation via session_id.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    doc_ids = _get_ready_doc_ids(request.doc_ids)
    if not doc_ids:
        raise HTTPException(status_code=404, detail="No processed documents available.")

    session_id = request.session_id or "default"
    history    = conversation_memory.get_context_for_llm(session_id)

    # Retrieve relevant chunks
    chunks = rag_pipeline.semantic_search(request.question, doc_ids, top_k=5)
    if not chunks:
        chunks = rag_pipeline.keyword_search(request.question, doc_ids, top_k=5)

    # Enrich chunks with doc names
    for chunk in chunks:
        doc = document_store.get(chunk["doc_id"])
        chunk["doc_name"] = doc["doc_name"] if doc else chunk["doc_id"]

    # Generate answer
    result = ai_service.answer_question(request.question, chunks, history)

    # Build citations
    citations = []
    seen = set()
    for chunk in chunks[:3]:
        key = f"{chunk['doc_id']}_{chunk['page_number']}"
        if key not in seen:
            citations.append(Citation(
                doc_id=chunk["doc_id"],
                doc_name=chunk.get("doc_name", chunk["doc_id"]),
                page_number=chunk["page_number"],
                excerpt=chunk["text"][:200] + "...",
            ))
            seen.add(key)

    # Update conversation memory
    conversation_memory.add_turn(session_id, "user", request.question)
    conversation_memory.add_turn(session_id, "assistant", result["answer"])

    # Track analytics
    analytics_tracker.record_question(doc_ids)

    logger.info(f"QA answered for session {session_id}")
    return QAResponse(
        question=request.question,
        answer=result["answer"],
        citations=citations,
        confidence_score=result["confidence_score"],
        session_id=session_id,
    )


# ── Summarization ─────────────────────────────────────────────────────────────

@router.post("/summarize", response_model=SummarizeResponse, summary="Summarize a document")
async def summarize_document(request: SummarizeRequest):
    """
    Generate document summaries. Types: executive | technical | bullet | key_takeaways
    """
    doc = document_store.get(request.doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if doc["processing_status"] != "ready":
        raise HTTPException(status_code=400, detail="Document is not ready for summarization.")

    valid_types = ["executive", "technical", "bullet", "key_takeaways"]
    if request.summary_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"summary_type must be one of: {valid_types}")

    full_text = rag_pipeline.get_full_text(request.doc_id)
    if not full_text:
        raise HTTPException(status_code=400, detail="No text content found for this document.")

    summary = ai_service.summarize(doc["doc_name"], full_text, request.summary_type)

    logger.info(f"Summarized {request.doc_id} as {request.summary_type}")
    return SummarizeResponse(
        doc_id=request.doc_id,
        doc_name=doc["doc_name"],
        summary_type=request.summary_type,
        summary=summary,
    )


# ── Document Comparison ───────────────────────────────────────────────────────

@router.post("/compare", response_model=CompareResponse, summary="Compare multiple documents")
async def compare_documents(request: CompareRequest):
    """
    Compare two or more documents on a specific aspect.
    Aspects: general | methodology | advantages | differences | similarities | conclusions
    """
    if len(request.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 document IDs required for comparison.")

    doc_names = []
    contexts  = []
    for doc_id in request.doc_ids:
        doc = document_store.get(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")
        if doc["processing_status"] != "ready":
            raise HTTPException(status_code=400, detail=f"Document {doc_id} is not ready.")
        doc_names.append(doc["doc_name"])

        # Get representative chunks for this document
        ctx = rag_pipeline.semantic_search(
            request.aspect or "general overview methodology findings",
            [doc_id],
            top_k=8
        )
        contexts.append(ctx)

    comparison = ai_service.compare_documents(doc_names, contexts, request.aspect or "general")

    return CompareResponse(
        doc_ids=request.doc_ids,
        doc_names=doc_names,
        aspect=request.aspect or "general",
        comparison=comparison,
    )


# ── Conversation Session Management ──────────────────────────────────────────

@router.get("/session/{session_id}", response_model=SessionResponse, summary="Get conversation history")
async def get_session(session_id: str):
    history = conversation_memory.get_history(session_id)
    return SessionResponse(session_id=session_id, history=history)


@router.delete("/session/{session_id}", response_model=ClearSessionResponse, summary="Clear conversation history")
async def clear_session(session_id: str):
    conversation_memory.clear_session(session_id)
    return ClearSessionResponse(session_id=session_id, message="Conversation history cleared.")
