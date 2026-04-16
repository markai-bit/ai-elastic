
from __future__ import annotations

from elasticsearch import AsyncElasticsearch

from app.config import get_settings

_client: AsyncElasticsearch | None = None


def build_es_client() -> AsyncElasticsearch:
    settings = get_settings()
    auth = None
    if settings.es_password:
        auth = (settings.es_username, settings.es_password)
    return AsyncElasticsearch(
        settings.es_url,
        basic_auth=auth,
        request_timeout=settings.request_timeout_seconds,
    )


async def get_es_client() -> AsyncElasticsearch:
    global _client
    if _client is None:
        _client = build_es_client()
    return _client


async def close_es_client() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
