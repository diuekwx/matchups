import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb


REQUIRED_FIELDS = {
    "champion",
    "opponent",
    "role",
    "source_url",
    "tip",
    "stats",
    "chunk_text",
    "created_at",
}

def load_chunks(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError("Expected the input JSON to contain a list")

    for index, chunk, in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {index} is not a JSON object")

        missing = REQUIRED_FIELDS - chunk.keys()

        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"Chunk {index} is missing fields: {fields}")

    return chunks

def compute_content_hash(chunk: dict[str, Any]) -> str:
    content = {
        "champion": chunk["champion"],
        "opponent": chunk["opponent"],
        "role": chunk["role"],
        "source_url": chunk["source_url"],
        "tip": chunk["tip"],
        "stats": chunk["stats"],
        "chunk_text": chunk["chunk_text"],
    }

    canonical_json = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

INSERT_CHUNK = """
    INSERT INTO matchup_chunks (
        champion,
        opponent,
        role,
        source_url,
        tip,
        stats,
        chunk_text,
        content_hash,
        scraped_at
    )
    VALUES (
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s
    )
    ON CONFLICT (
        champion,
        opponent,
        role,
        source_url,
        content_hash
    )
    DO NOTHING
    RETURNING id
"""

def ingest_chunks(
    database_url: str,
    chunks: list[dict[str, Any]],
) -> tuple[int, int]:
    inserted = 0
    skipped = 0

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            for chunk in chunks:
                cursor.execute(
                    INSERT_CHUNK,
                    (
                        chunk["champion"],
                        chunk["opponent"],
                        chunk["role"],
                        chunk["source_url"],
                        chunk["tip"],
                        Jsonb(chunk["stats"]),
                        chunk["chunk_text"],
                        compute_content_hash(chunk),
                        parse_timestamp(chunk["created_at"]),
                    ),
                )

                if cursor.fetchone() is None:
                    skipped += 1
                else:
                    inserted += 1

    return inserted, skipped

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest scraped matchup chunks into PostgreSQL"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the scraper's chunks JSON file",
    )
    args = parser.parse_args()

    load_dotenv(override=True)

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    chunks = load_chunks(args.input)
    inserted, skipped = ingest_chunks(database_url, chunks)

    print(f"Read:     {len(chunks)}")
    print(f"Inserted: {inserted}")
    print(f"Skipped:  {skipped}")


if __name__ == "__main__":
    main()