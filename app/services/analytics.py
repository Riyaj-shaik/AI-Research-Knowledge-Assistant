"""
analytics.py - Tracks usage metrics across the application.
"""

import json
import os
from collections import defaultdict
from app.core.config import settings
from app.core.logging import logger


class AnalyticsTracker:

    def __init__(self):
        self._file = settings.ANALYTICS_FILE
        os.makedirs(os.path.dirname(self._file), exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._file):
            try:
                with open(self._file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "total_questions_answered": 0,
            "document_query_counts": {},
        }

    def _save(self):
        with open(self._file, "w") as f:
            json.dump(self._data, f, indent=2)

    def record_question(self, doc_ids: list):
        self._data["total_questions_answered"] += 1
        for doc_id in (doc_ids or []):
            self._data["document_query_counts"][doc_id] = (
                self._data["document_query_counts"].get(doc_id, 0) + 1
            )
        self._save()

    def get_stats(self) -> dict:
        return self._data

    def get_most_queried(self, top_n: int = 5) -> list:
        counts = self._data.get("document_query_counts", {})
        sorted_docs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [{"doc_id": k, "query_count": v} for k, v in sorted_docs[:top_n]]


# Singleton
analytics_tracker = AnalyticsTracker()
