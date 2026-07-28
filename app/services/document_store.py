"""
document_store.py - Manages document metadata persistence and retrieval.
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional, List

from app.core.config import settings
from app.core.logging import logger


class DocumentStore:
    """
    Lightweight JSON-backed metadata store for uploaded documents.
    """

    def __init__(self):
        self._file = settings.METADATA_FILE
        os.makedirs(os.path.dirname(self._file), exist_ok=True)
        self._data: Dict[str, dict] = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._file):
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata store: {e}")
        return {}

    def _save(self):
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def add(self, doc_id: str, metadata: dict):
        self._data[doc_id] = metadata
        self._save()
        logger.info(f"Document registered: {doc_id} — {metadata.get('doc_name')}")

    def get(self, doc_id: str) -> Optional[dict]:
        return self._data.get(doc_id)

    def all(self) -> List[dict]:
        return list(self._data.values())

    def update(self, doc_id: str, fields: dict):
        if doc_id in self._data:
            self._data[doc_id].update(fields)
            self._save()

    def delete(self, doc_id: str) -> bool:
        if doc_id in self._data:
            del self._data[doc_id]
            self._save()
            logger.info(f"Document deleted from store: {doc_id}")
            return True
        return False

    def exists(self, doc_id: str) -> bool:
        return doc_id in self._data

    def count(self) -> int:
        return len(self._data)


# Singleton instance
document_store = DocumentStore()
