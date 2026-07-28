"""
documents.py - REST endpoints for document upload, listing, and deletion.
"""

import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import DocumentMetadata, DocumentListResponse, DeleteResponse
from app.services.document_store import document_store
from app.services.rag_pipeline import rag_pipeline
from app.ml.classifier import document_classifier

router = APIRouter(prefix="/documents", tags=["Document Management"])


def process_document_background(doc_id: str, filepath: str, doc_name: str):
    """Background task: extract → chunk → embed → index → classify."""
    try:
        document_store.update(doc_id, {"processing_status": "processing"})

        # 1. Extract text
        pages = rag_pipeline.extract_text_from_pdf(filepath)
        if not pages:
            raise ValueError("No text could be extracted from the PDF.")

        # 2. Chunk
        chunks = rag_pipeline.chunk_pages(pages)

        # 3. Embed & index
        total_chunks = rag_pipeline.build_index(doc_id, chunks)

        # 4. Classify (if model is trained)
        category = None
        confidence = None
        if document_classifier.is_trained():
            full_text = " ".join(p["text"] for p in pages)
            result    = document_classifier.predict(full_text)
            category  = result["category"]
            confidence = result["confidence"]

        document_store.update(doc_id, {
            "total_pages": len(pages),
            "total_chunks": total_chunks,
            "processing_status": "ready",
            "category": category,
            "category_confidence": confidence,
        })
        logger.info(f"Document {doc_id} processed successfully.")

    except Exception as e:
        document_store.update(doc_id, {"processing_status": "failed"})
        logger.error(f"Processing failed for {doc_id}: {e}")


@router.post("/upload", response_model=DocumentMetadata, summary="Upload a PDF document")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF document. Processing (chunking, embedding, indexing) runs in the background.
    Poll GET /documents/{doc_id} to check processing status.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_FILE_SIZE_MB} MB limit.")

    doc_id   = str(uuid.uuid4())
    doc_name = file.filename
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(settings.UPLOAD_DIR, f"{doc_id}.pdf")

    with open(filepath, "wb") as f:
        f.write(content)

    metadata = {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "upload_timestamp": datetime.utcnow().isoformat() + "Z",
        "total_pages": 0,
        "total_chunks": 0,
        "processing_status": "pending",
        "file_size_kb": round(size_mb * 1024, 2),
        "category": None,
        "category_confidence": None,
    }
    document_store.add(doc_id, metadata)

    background_tasks.add_task(process_document_background, doc_id, filepath, doc_name)
    logger.info(f"Uploaded document: {doc_name} → {doc_id}")

    return DocumentMetadata(**metadata)


@router.get("/", response_model=DocumentListResponse, summary="List all documents")
async def list_documents():
    docs = document_store.all()
    return DocumentListResponse(total=len(docs), documents=[DocumentMetadata(**d) for d in docs])


@router.get("/{doc_id}", response_model=DocumentMetadata, summary="Get document details")
async def get_document(doc_id: str):
    doc = document_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentMetadata(**doc)


@router.delete("/{doc_id}", response_model=DeleteResponse, summary="Delete a document")
async def delete_document(doc_id: str):
    if not document_store.exists(doc_id):
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove files
    filepath = os.path.join(settings.UPLOAD_DIR, f"{doc_id}.pdf")
    if os.path.exists(filepath):
        os.remove(filepath)

    rag_pipeline.delete_index(doc_id)
    document_store.delete(doc_id)

    return DeleteResponse(doc_id=doc_id, message="Document deleted successfully.")


@router.post("/{doc_id}/reprocess", response_model=DocumentMetadata, summary="Reprocess a document")
async def reprocess_document(doc_id: str, background_tasks: BackgroundTasks):
    doc = document_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    filepath = os.path.join(settings.UPLOAD_DIR, f"{doc_id}.pdf")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Original PDF file not found.")

    rag_pipeline.delete_index(doc_id)
    document_store.update(doc_id, {"processing_status": "pending", "total_chunks": 0})
    background_tasks.add_task(process_document_background, doc_id, filepath, doc["doc_name"])

    return DocumentMetadata(**document_store.get(doc_id))
