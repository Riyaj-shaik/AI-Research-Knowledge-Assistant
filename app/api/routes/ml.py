"""
ml.py - TensorFlow model training and document classification endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.models.schemas import ClassifyResponse
from app.ml.classifier import document_classifier
from app.services.document_store import document_store
from app.services.rag_pipeline import rag_pipeline
from app.core.logging import logger

router = APIRouter(prefix="/ml", tags=["ML Classification"])


@router.post("/train", summary="Train the TensorFlow document classifier")
async def train_classifier(background_tasks: BackgroundTasks):
    """
    Train the TensorFlow text classification model on the built-in labeled dataset.
    Training runs in the background. Check /ml/status to monitor.
    """
    def _train():
        try:
            metrics = document_classifier.train()
            logger.info(f"Training complete: {metrics}")
        except Exception as e:
            logger.error(f"Training failed: {e}")

    background_tasks.add_task(_train)
    return {"message": "Model training started in background. Check /ml/status for progress."}


@router.get("/status", summary="Check classifier training status")
async def classifier_status():
    return {
        "is_trained": document_classifier.is_trained(),
        "model_file": "data/models/classifier.keras",
        "categories": document_classifier.categories,
        "vocab_size": len(document_classifier.word_index),
    }


@router.post("/classify/{doc_id}", response_model=ClassifyResponse, summary="Classify a document")
async def classify_document(doc_id: str):
    """
    Classify a document into one of the predefined categories using the trained TF model.
    """
    if not document_classifier.is_trained():
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet. POST /ml/train first."
        )

    doc = document_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    if doc["processing_status"] != "ready":
        raise HTTPException(status_code=400, detail="Document is not ready yet.")

    full_text = rag_pipeline.get_full_text(doc_id)
    if not full_text:
        raise HTTPException(status_code=400, detail="No text content found.")

    result = document_classifier.predict(full_text)

    # Persist classification back to document metadata
    from app.services.document_store import document_store as ds
    ds.update(doc_id, {
        "category": result["category"],
        "category_confidence": result["confidence"],
    })

    return ClassifyResponse(
        doc_id=doc_id,
        doc_name=doc["doc_name"],
        category=result["category"],
        confidence=result["confidence"],
        all_scores=result["all_scores"],
    )


@router.post("/classify-text", summary="Classify raw text")
async def classify_text(payload: dict):
    """
    Classify arbitrary text (not necessarily an uploaded document).
    Body: {"text": "your text here"}
    """
    if not document_classifier.is_trained():
        raise HTTPException(status_code=400, detail="Model not trained. POST /ml/train first.")

    text = payload.get("text", "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="text field is required.")

    result = document_classifier.predict(text)
    return result
