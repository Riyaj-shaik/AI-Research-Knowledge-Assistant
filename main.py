"""
main.py - FastAPI application entry point.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Swagger UI: http://localhost:8000/docs
ReDoc:      http://localhost:8000/redoc
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, search, assistant, ml, analytics
from app.core.logging import logger

app = FastAPI(
    title="AI Research & Knowledge Assistant",
    description=(
        "A production-ready RAG-powered API for intelligent document management, "
        "semantic search, question answering with citations, document comparison, "
        "summarization, and TensorFlow document classification."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(assistant.router)
app.include_router(ml.router)
app.include_router(analytics.router)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "running",
        "message": "AI Research & Knowledge Assistant API",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health", tags=["Health"])
async def health():
    from app.services.document_store import document_store
    from app.ml.classifier import document_classifier
    return {
        "status": "healthy",
        "documents_in_store": document_store.count(),
        "classifier_trained": document_classifier.is_trained(),
    }


logger.info("AI Research Assistant API started.")
