import sqlite3
from pathlib import Path

import pytest

from app.utils import zep
from app.utils.local_graph import LocalGraphClient, NotFoundError


def test_permanent_errors_fail_without_retry():
    calls = []

    def operation():
        calls.append(True)
        raise NotFoundError("missing")

    with pytest.raises(NotFoundError):
        zep.call_zep_read_with_retry(
            operation,
            operation_name="permanent failure",
            sleep=lambda _seconds: None,
        )

    assert len(calls) == 1


def test_locked_database_read_is_retried():
    calls = []
    sleeps = []

    def operation():
        calls.append(True)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return "ok"

    result = zep.call_zep_read_with_retry(
        operation,
        operation_name="locked read",
        initial_delay=0.5,
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert len(calls) == 2
    assert sleeps == [0.5]


def test_graph_client_is_shared_per_database_file(tmp_path):
    zep.clear_zep_client_cache()
    db_path = str(tmp_path / "graph.sqlite3")

    first = zep.get_zep_client(db_path)
    second = zep.get_zep_client(db_path)
    other = zep.get_zep_client(str(tmp_path / "other.sqlite3"))

    assert isinstance(first, LocalGraphClient)
    assert first is second
    assert other is not first
    zep.clear_zep_client_cache()


def test_env_example_has_no_graph_service_settings():
    env_example = Path(__file__).resolve().parents[2] / ".env.example"
    contents = env_example.read_text(encoding="utf-8")

    assert "ZEP_API_KEY" not in contents
    assert "ZEP_API_URL" not in contents
