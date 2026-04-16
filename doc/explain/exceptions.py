
from __future__ import annotations


class RagError(Exception):
    """Base class for all RAG service errors."""

    http_status: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class MetadataValidationError(RagError):
    http_status = 422
    error_code = "metadata_validation_failed"


class EmbeddingError(RagError):
    http_status = 502
    error_code = "embedding_service_error"


class IndexNotFoundError(RagError):
    http_status = 404
    error_code = "index_not_found"


class IndexOperationError(RagError):
    http_status = 502
    error_code = "index_operation_failed"


class SearchError(RagError):
    http_status = 502
    error_code = "search_failed"


class DuplicateContentError(RagError):
    """Surface dedup outcomes if the caller wants 409 instead of 200/skipped."""
    http_status = 409
    error_code = "duplicate_content"
