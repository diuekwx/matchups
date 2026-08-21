import argparse
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
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
VALID_ROLES = {"top", "jungle", "mid", "adc", "support"}
MAX_RESULTS = 50


def require_environment_variable(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def embed_query(client: genai.Client, question: str) -> list[float]:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type=QUERY_TASK_TYPE,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )

    if not response.embeddings:
        raise ValueError("Gemini returned no query embedding")

    embedding = response.embeddings[0].values
    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS} dimensions, "
            f"received {len(embedding)}"
        )

    return embedding


def retrieve_matches(
    connection: psycopg.Connection,
    query_embedding: list[float],
    champion: str,
    opponent: str,
    role: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    champion = champion.strip()
    opponent = opponent.strip()
    role = role.strip().lower()

    if not champion or not opponent:
        raise ValueError("Champion and opponent cannot be empty")
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role: {role}")
    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"Limit must be between 1 and {MAX_RESULTS}")
    if len(query_embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS} query dimensions, "
            f"received {len(query_embedding)}"
        )

    query_vector = Vector(query_embedding)
    cursor = connection.execute(
        """
        SELECT
            id,
            champion,
            opponent,
            role,
            source_url,
            chunk_text,
            embedding <=> %s AS distance
        FROM matchup_chunks
        WHERE champion = %s
          AND opponent = %s
          AND role = %s
          AND embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (
            query_vector,
            champion,
            opponent,
            role,
            query_vector,
            limit,
        ),
    )
    return cursor.fetchall()


def print_results(results: list[dict[str, Any]]) -> None:
    if not results:
        print("No matching embedded chunks found")
        return

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['champion']} vs {result['opponent']} ({result['role']})")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Source: {result['source_url']}")
        print("Chunk:")
        print(result["chunk_text"])
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve matchup chunks using metadata and vector similarity"
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--role", required=True, choices=sorted(VALID_ROLES))
    parser.add_argument("--limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()

    database_url = require_environment_variable("DATABASE_URL")
    gemini_api_key = require_environment_variable("GEMINI_API_KEY")
    client = genai.Client(api_key=gemini_api_key)
    query_embedding = embed_query(client, args.question)

    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        register_vector(connection)
        results = retrieve_matches(
            connection=connection,
            query_embedding=query_embedding,
            champion=args.champion,
            opponent=args.opponent,
            role=args.role,
            limit=args.limit,
        )

    print_results(results)


if __name__ == "__main__":
    main()
