from types import SimpleNamespace

import pytest
from pgvector import Vector

from retrieve import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    QUERY_TASK_TYPE,
    embed_query,
    print_results,
    retrieve_matches,
)


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


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executions = []

    def execute(self, query, parameters):
        self.executions.append((query, parameters))
        return FakeCursor(self.rows)


def test_embed_query_returns_768_values_with_query_configuration():
    values = [0.02] * EMBEDDING_DIMENSIONS
    client = FakeClient([SimpleNamespace(values=values)])

    assert embed_query(client, "  How do I beat Yone?  ") == values

    [call] = client.models.calls
    assert call["model"] == EMBEDDING_MODEL
    assert call["contents"] == "How do I beat Yone?"
    assert call["config"].task_type == QUERY_TASK_TYPE
    assert call["config"].output_dimensionality == EMBEDDING_DIMENSIONS


def test_embed_query_rejects_empty_question_without_api_call():
    client = FakeClient([SimpleNamespace(values=[0.02] * EMBEDDING_DIMENSIONS)])

    with pytest.raises(ValueError, match="Question cannot be empty"):
        embed_query(client, "   ")

    assert client.models.calls == []


def test_embed_query_rejects_missing_response():
    with pytest.raises(ValueError, match="Gemini returned no query embedding"):
        embed_query(FakeClient([]), "How do I beat Yone?")


def test_embed_query_rejects_wrong_dimensions():
    client = FakeClient([SimpleNamespace(values=[0.02] * 767)])

    with pytest.raises(ValueError, match="Expected 768 dimensions, received 767"):
        embed_query(client, "How do I beat Yone?")


def test_retrieve_matches_filters_metadata_and_returns_rows():
    expected = [
        {
            "id": 1,
            "champion": "Malphite",
            "opponent": "Yone",
            "role": "top",
            "source_url": "https://op.gg/example",
            "chunk_text": "Matchup advice",
            "distance": 0.12,
        }
    ]
    connection = FakeConnection(expected)
    embedding = [0.02] * EMBEDDING_DIMENSIONS

    results = retrieve_matches(
        connection,
        embedding,
        champion=" Malphite ",
        opponent=" Yone ",
        role="TOP",
        limit=3,
    )

    assert results == expected
    [(query, parameters)] = connection.executions
    selected_vector, champion, opponent, role, ordered_vector, limit = parameters
    assert "embedding IS NOT NULL" in query
    assert "ORDER BY embedding <=> %s" in query
    assert isinstance(selected_vector, Vector)
    assert selected_vector.to_list() == pytest.approx(embedding)
    assert ordered_vector.to_list() == pytest.approx(embedding)
    assert (champion, opponent, role, limit) == ("Malphite", "Yone", "top", 3)


@pytest.mark.parametrize("limit", [0, 51])
def test_retrieve_matches_rejects_invalid_limit(limit):
    with pytest.raises(ValueError, match="Limit must be between 1 and 50"):
        retrieve_matches(
            FakeConnection(),
            [0.02] * EMBEDDING_DIMENSIONS,
            champion="Malphite",
            opponent="Yone",
            role="top",
            limit=limit,
        )


def test_retrieve_matches_rejects_invalid_role():
    with pytest.raises(ValueError, match="Invalid role: bottom"):
        retrieve_matches(
            FakeConnection(),
            [0.02] * EMBEDDING_DIMENSIONS,
            champion="Malphite",
            opponent="Yone",
            role="bottom",
        )


def test_retrieve_matches_rejects_wrong_query_dimensions():
    with pytest.raises(ValueError, match="Expected 768 query dimensions"):
        retrieve_matches(
            FakeConnection(),
            [0.02] * 767,
            champion="Malphite",
            opponent="Yone",
            role="top",
        )


def test_print_results_handles_no_matches(capsys):
    print_results([])

    assert capsys.readouterr().out.strip() == "No matching embedded chunks found"
