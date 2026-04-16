"""
HTTP cross-cutting concerns:
  - Correlation ID middleware (read X-Correlation-ID or mint one)
  - Access logging
  - Exception handlers that translate RagError -> JSON error responses
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.exceptions import RagError
from app.logging_config import get_correlation_id, get_logger, set_correlation_id
from app.models import ErrorResponse

log = get_logger("app.http")

CORRELATION_HEADER = "X-Correlation-ID"


def install_middleware_and_handlers(app: FastAPI) -> None:
    @app.middleware("http")
    async def correlation_and_access_log(request: Request, call_next):
        incoming = request.headers.get(CORRELATION_HEADER)
        cid = set_correlation_id(incoming)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            log.exception(
                "request_failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers[CORRELATION_HEADER] = cid
        log.info(
            "request_completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response

    @app.exception_handler(RagError)
    async def handle_rag_error(_: Request, exc: RagError) -> JSONResponse:
        log.warning(
            "domain_error",
            extra={"error_code": exc.error_code, "details": exc.details, "message": exc.message},
        )
        body = ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            correlation_id=get_correlation_id(),
        )
        return JSONResponse(status_code=exc.http_status, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        body = ErrorResponse(
            error_code="request_validation_failed",
            message="Request body failed validation",
            details={"errors": exc.errors()},
            correlation_id=get_correlation_id(),
        )
        log.warning("validation_error", extra={"errors": exc.errors()})
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled_exception")
        body = ErrorResponse(
            error_code="internal_error",
            message="Internal server error",
            details={"type": exc.__class__.__name__},
            correlation_id=get_correlation_id(),
        )
        return JSONResponse(status_code=500, content=body.model_dump())
