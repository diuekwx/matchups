from types import SimpleNamespace

import pytest
from google.genai import errors
from pgvector import Vector

from embedding import (
    DOCUMENT_TASK_TYPE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    embed_row,
    embed_row_with_retry,
    require_environment_variable,
    retrieve_rows,
    retry_delay_seconds,
    update_embedding,
)


def make_row(**overrides):
    row = {
        "id": 7,
        "champion": "Malphite",
        "opponent": "Yone",
        "role": "top",
        "chunk_text": "Use E to reduce Yone's attack speed.",
    }
    row.update(overrides)
    return row


class FakeModels:
    def __init__(self, embeddings):
        self.response = SimpleNamespace(embeddings=embeddings)
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, embeddings):
        self.models = FakeModels(embeddings)


class SequencedModels:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def embed_content(self, **_kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(embeddings=outcome)


class SequencedClient:
    def __init__(self, outcomes):
        self.models = SequencedModels(outcomes)


class FakeResult:
    def __init__(self, rowcount):
        self.rowcount = rowcount


class FakeConnection:
    def __init__(self, rowcount=1):
        self.rowcount = rowcount
        self.executions = []

    def execute(self, query, parameters):
        self.executions.append((query, parameters))
        return FakeResult(self.rowcount)


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query):
        self.query = query

    def fetchall(self):
        return self.rows


class FakeReadConnection:
    def __init__(self, rows):
        self.fake_cursor = FakeCursor(rows)

    def cursor(self):
        return self.fake_cursor


def test_require_environment_variable_returns_value(monkeypatch):
    monkeypatch.setenv("EXAMPLE_SETTING", "configured")

    assert require_environment_variable("EXAMPLE_SETTING") == "configured"


def test_require_environment_variable_rejects_missing_value(monkeypatch):
    monkeypatch.delenv("EXAMPLE_SETTING", raising=False)

    with pytest.raises(RuntimeError, match="EXAMPLE_SETTING is not set"):
        require_environment_variable("EXAMPLE_SETTING")


def test_retrieve_rows_returns_cursor_results_and_filters_pending_rows():
    expected = [make_row()]
    connection = FakeReadConnection(expected)

    assert retrieve_rows(connection) == expected
    assert "WHERE embedding IS NULL" in connection.fake_cursor.query


def test_embed_row_returns_768_values_and_uses_document_configuration():
    values = [0.01] * EMBEDDING_DIMENSIONS
    client = FakeClient([SimpleNamespace(values=values)])
    row = make_row()

    assert embed_row(client, row) == values

    [call] = client.models.calls
    assert call["model"] == EMBEDDING_MODEL
    assert call["contents"] == row["chunk_text"]
    assert call["config"].task_type == DOCUMENT_TASK_TYPE
    assert call["config"].output_dimensionality == EMBEDDING_DIMENSIONS


def test_embed_row_rejects_empty_text_without_calling_gemini():
    client = FakeClient([SimpleNamespace(values=[0.01] * EMBEDDING_DIMENSIONS)])

    with pytest.raises(ValueError, match="Chunk 7 has empty chunk_text"):
        embed_row(client, make_row(chunk_text="   "))

    assert client.models.calls == []


def test_embed_row_rejects_missing_embedding_response():
    client = FakeClient([])

    with pytest.raises(ValueError, match="Gemini returned no embedding"):
        embed_row(client, make_row())


def test_embed_row_rejects_wrong_dimensions():
    client = FakeClient([SimpleNamespace(values=[0.01] * 767)])

    with pytest.raises(ValueError, match="Expected 768 dimensions, received 767"):
        embed_row(client, make_row())


def test_retry_delay_uses_gemini_retry_info_with_buffer():
    error = errors.ClientError(429, {
        "error": {"details": [{"retryDelay": "21.087s"}]}
    })

    assert retry_delay_seconds(error, attempt=1) == pytest.approx(22.087)


def test_embed_row_with_retry_recovers_from_rate_limit():
    rate_limit = errors.ClientError(429, {
        "error": {"details": [{"retryDelay": "2s"}]}
    })
    values = [0.01] * EMBEDDING_DIMENSIONS
    client = SequencedClient([rate_limit, [SimpleNamespace(values=values)]])
    delays = []

    result = embed_row_with_retry(
        client,
        make_row(),
        max_retries=2,
        sleep=delays.append,
    )

    assert result == values
    assert client.models.calls == 2
    assert delays == [3.0]


def test_embed_row_with_retry_does_not_retry_nonretryable_error():
    bad_request = errors.ClientError(400, {"error": {"message": "bad request"}})
    client = SequencedClient([bad_request])
    delays = []

    with pytest.raises(errors.ClientError):
        embed_row_with_retry(
            client,
            make_row(),
            max_retries=2,
            sleep=delays.append,
        )

    assert client.models.calls == 1
    assert delays == []


def test_update_embedding_stores_vector_for_matching_row():
    connection = FakeConnection()
    values = [0.01] * EMBEDDING_DIMENSIONS

    update_embedding(connection, row_id=7, embedding=values)

    [(query, parameters)] = connection.executions
    stored_vector, stored_row_id = parameters
    assert "SET embedding = %s" in query
    assert "AND embedding IS NULL" in query
    assert isinstance(stored_vector, Vector)
    assert stored_vector.to_list() == pytest.approx(values)
    assert stored_row_id == 7


def test_update_embedding_rejects_row_that_was_not_updated():
    connection = FakeConnection(rowcount=0)

    with pytest.raises(RuntimeError, match="Chunk 7 was not updated"):
        update_embedding(
            connection,
            row_id=7,
            embedding=[0.01] * EMBEDDING_DIMENSIONS,
        )
