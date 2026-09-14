import argparse
import os
import re
import time
from typing import Any

import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row


EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
DEFAULT_REQUEST_DELAY = 1.0
DEFAULT_MAX_RETRIES = 8
DEFAULT_BACKOFF_SECONDS = 5.0
MAX_BACKOFF_SECONDS = 60.0


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


def retry_delay_seconds(error: errors.APIError, attempt: int) -> float:
    """Use Gemini's RetryInfo when present, otherwise exponential backoff."""
    details = error.details
    if isinstance(details, dict):
        error_details = details.get("error", {}).get("details", [])
        for detail in error_details:
            if not isinstance(detail, dict):
                continue
            retry_delay = detail.get("retryDelay")
            if isinstance(retry_delay, str):
                match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)s", retry_delay)
                if match:
                    # A small buffer avoids retrying on the quota boundary.
                    return float(match.group(1)) + 1.0

    return min(
        DEFAULT_BACKOFF_SECONDS * (2 ** (attempt - 1)),
        MAX_BACKOFF_SECONDS,
    )


def embed_row_with_retry(
    client: genai.Client,
    row: dict[str, Any],
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep: Any = time.sleep,
) -> list[float]:
    """Embed one row, retrying rate limits and transient server failures."""
    for attempt in range(1, max_retries + 2):
        try:
            return embed_row(client, row)
        except errors.APIError as error:
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt > max_retries:
                raise

            delay = retry_delay_seconds(error, attempt)
            print(
                f"Chunk {row['id']}: Gemini returned {error.code}; "
                f"retrying in {delay:.1f}s "
                f"(attempt {attempt}/{max_retries})"
            )
            sleep(delay)

    raise AssertionError("retry loop exited unexpectedly")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed pending matchup chunks")
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help="Seconds between successful requests (default: 1.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries per chunk for HTTP 429 and 5xx errors (default: 8)",
    )
    args = parser.parse_args()
    if args.request_delay < 0:
        parser.error("--request-delay cannot be negative")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    return args


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()

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
            embedding = embed_row_with_retry(
                client,
                row,
                max_retries=args.max_retries,
            )
            update_embedding(connection, row["id"], embedding)
            connection.commit()

            print(
                f"[{index}/{len(rows)}] Embedded "
                f"{row['champion']} vs {row['opponent']} ({row['role']})"
            )
            if args.request_delay and index < len(rows):
                time.sleep(args.request_delay)

    print(f"Embedded {len(rows)} chunks")


if __name__ == "__main__":
    main()
