

from __future__ import annotations

import hashlib
import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse all runs of whitespace to a single space and strip."""
    if text is None:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def content_hash(text: str) -> str:
    """SHA-256 hex digest of normalized text. Stable, 64 chars."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
