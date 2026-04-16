"""
FastAPI application entrypoint.

Run locally:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.es_client import close_es_client
from app.logging_config import configure_logging, get_logger
from app.middleware import install_middleware_and_handlers
from app.routers.ingestion_router import router as ingestion_router
from app.routers.search_router import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(level=settings.log_level)
    log = get_logger("app.main")
    log.info("service_starting", extra={"index": settings.index_name, "es_url": settings.es_url})
    yield
    await close_es_client()
    log.info("service_stopped")


app = FastAPI(
    title="RAG Service (POC2)",
    description="Ingestion + hybrid search over Elasticsearch with semantic routing.",
    version="0.1.0",
    lifespan=lifespan,
)

install_middleware_and_handlers(app)
app.include_router(ingestion_router)
app.include_router(search_router)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
