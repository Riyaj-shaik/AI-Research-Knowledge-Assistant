"""
config.py - Centralised application configuration loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── Gemini ────────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str        = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str          = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
    EMBEDDING_MODEL: str       = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

    # ── Storage ───────────────────────────────────────────────────────────────
    UPLOAD_DIR: str            = os.getenv("UPLOAD_DIR", "data/uploads")
    INDEX_DIR: str             = os.getenv("INDEX_DIR", "data/indexes")
    MODEL_DIR: str             = os.getenv("MODEL_DIR", "data/models")
    METADATA_FILE: str         = "data/metadata.json"
    ANALYTICS_FILE: str        = "data/analytics.json"

    # ── Processing ────────────────────────────────────────────────────────────
    CHUNK_SIZE: int            = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int         = int(os.getenv("CHUNK_OVERLAP", 100))
    TOP_K_RESULTS: int         = int(os.getenv("TOP_K_RESULTS", 5))
    MAX_FILE_SIZE_MB: int      = int(os.getenv("MAX_FILE_SIZE_MB", 50))

    # ── ML Classification ─────────────────────────────────────────────────────
    CATEGORIES: list           = [
        "Artificial Intelligence",
        "Machine Learning",
        "Computer Vision",
        "Natural Language Processing",
        "Robotics",
        "Cyber Security",
        "Cloud Computing",
    ]
    MAX_VOCAB_SIZE: int        = 10000
    MAX_SEQUENCE_LENGTH: int   = 500
    ML_MODEL_FILE: str         = "data/models/classifier.keras"
    TOKENIZER_FILE: str        = "data/models/tokenizer.json"

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str             = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
