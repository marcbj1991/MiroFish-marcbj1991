"""Shared local graph client, request limits, and read retry policy.

The module name is kept from the former Zep Cloud integration; the client it
returns is the local SQLite store in ``local_graph.py``.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import Any, Callable, TypeVar

from ..config import Config
from .local_graph import LocalGraphClient
from .logger import get_logger

logger = get_logger("mirofish.zep")

T = TypeVar("T")

# Upper bound for waiting on graph ingestion (LLM extraction of every chunk
# runs through the configured model and can take a long time).
ZEP_INGESTION_WAIT_TIMEOUT_SECONDS = 3600
# Allowance for one in-flight graph write (one LLM extraction request; the
# Anthropic SDK default request timeout is 10 minutes).
ZEP_HTTP_REQUEST_TIMEOUT_SECONDS = 600.0
MAX_ZEP_SEARCH_QUERY_CHARS = 400
MAX_ZEP_SEARCH_RESULTS = 50

_clients: dict[str, LocalGraphClient] = {}
_clients_guard = threading.Lock()


def normalize_zep_search_query(query: Any) -> str:
    """Return a non-empty, length-capped search query."""

    if not isinstance(query, str):
        raise ValueError("Zep search query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("Zep search query must not be empty")
    return normalized[:MAX_ZEP_SEARCH_QUERY_CHARS]


def normalize_zep_search_limit(limit: Any) -> int:
    """Clamp a search result limit."""

    try:
        normalized = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("Zep search limit must be an integer") from exc
    if normalized < 1:
        raise ValueError("Zep search limit must be at least 1")
    return min(normalized, MAX_ZEP_SEARCH_RESULTS)


def get_zep_client(db_path: str | None = None) -> LocalGraphClient:
    """Return the process-shared local graph client for ``db_path``."""

    path = os.path.abspath(db_path or Config.GRAPH_DB_PATH)
    with _clients_guard:
        client = _clients.get(path)
        if client is None:
            client = LocalGraphClient(path)
            _clients[path] = client
        return client


def clear_zep_client_cache() -> None:
    """Clear cached clients. Intended for tests and controlled reconfiguration."""

    with _clients_guard:
        _clients.clear()


def is_retryable_zep_error(error: BaseException) -> bool:
    """Return whether a failed *read* is safe and useful to retry."""

    if isinstance(error, sqlite3.OperationalError):
        message = str(error).lower()
        return "locked" in message or "busy" in message
    return False


def call_zep_read_with_retry(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry a safe read only when the SQLite database is busy/locked."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == max_attempts or not is_retryable_zep_error(error):
                raise

            delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "Graph %s attempt %s/%s failed (%s); retrying in %.1fs",
                operation_name,
                attempt,
                max_attempts,
                type(error).__name__,
                delay,
            )
            sleep(delay)

    raise AssertionError("unreachable")
