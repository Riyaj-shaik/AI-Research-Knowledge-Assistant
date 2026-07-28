"""
logging.py - Centralised logging configuration.
"""

import logging
import sys
from app.core.config import settings


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("ai_research_assistant")


logger = setup_logging()
