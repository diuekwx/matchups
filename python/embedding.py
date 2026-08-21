import os
from typing import Any

import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"


def require_environment_variable(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def retrieve_rows(connection: psycopg.Connection) -> list[dict[str, Any]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, champion, opponent, role, chunk_text
            FROM matchup_chunks
            WHERE embedding IS NULL
            ORDER BY id
            """
        )
        return cursor.fetchall()


def embed_row(client: genai.Client, row: dict[str, Any]) -> list[float]:
    chunk_text = row["chunk_text"]
    if not chunk_text.strip():
        raise ValueError(f"Chunk {row['id']} has empty chunk_text")

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=chunk_text,
        config=types.EmbedContentConfig(
            task_type=DOCUMENT_TASK_TYPE,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )

    if not response.embeddings:
        raise ValueError(f"Gemini returned no embedding for chunk {row['id']}")

    embedding = response.embeddings[0].values
    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS} dimensions, "
            f"received {len(embedding)} for chunk {row['id']}"
        )

    return embedding


def update_embedding(
    connection: psycopg.Connection,
    row_id: int,
    embedding: list[float],
) -> None:
    result = connection.execute(
        """
        UPDATE matchup_chunks
        SET embedding = %s
        WHERE id = %s
          AND embedding IS NULL
        """,
        (Vector(embedding), row_id),
    )
    if result.rowcount != 1:
        raise RuntimeError(
            f"Chunk {row_id} was not updated; it may already be embedded"
        )


def main() -> None:
    load_dotenv(override=True)

    database_url = require_environment_variable("DATABASE_URL")
    gemini_api_key = require_environment_variable("GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)

    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        register_vector(connection)
        rows = retrieve_rows(connection)

        if not rows:
            print("No chunks awaiting embeddings")
            return

        print(f"Found {len(rows)} chunks awaiting embeddings")

        for index, row in enumerate(rows, start=1):
            embedding = embed_row(client, row)
            update_embedding(connection, row["id"], embedding)
            connection.commit()

            print(
                f"[{index}/{len(rows)}] Embedded "
                f"{row['champion']} vs {row['opponent']} ({row['role']})"
            )

    print(f"Embedded {len(rows)} chunks")


if __name__ == "__main__":
    main()
