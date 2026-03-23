# src/app/config/__init__.py
"""
Central configuration module.
Imports and exposes all config components for easy access.
Usage: from app.config import QDRANT_HOST, APP_ENV, get_logger, etc.
"""

from .settings import (
    APP_ENV,
    TSPEC_DATA,
    CHUNKS_FILE,
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTION_NAME,
    NUMBER_RETRIEVE_CHUNKS,
    OPENAI_API_KEY,
    MODEL,
    EMBEDDING_MODEL,
)
from .logging_config import log_level

_logging_configured = False

def get_logger(name):
    """
    Configures logging globally (if not already done) and returns
    a logger with the given name.

    Usage:
        logger = get_logger(__name__)
    """
    global _logging_configured
    if not _logging_configured:
        from . import logging_config
        _logging_configured = True

    import logging
    return logging.getLogger(name)


__all__ = [
    "APP_ENV",
    "TSPEC_DATA",
    "CHUNKS_FILE",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "COLLECTION_NAME",
    "NUMBER_RETRIEVE_CHUNKS",
    "OPENAI_API_KEY",
    "MODEL",
    "EMBEDDING_MODEL",
    "log_level",
    "get_logger",
]
