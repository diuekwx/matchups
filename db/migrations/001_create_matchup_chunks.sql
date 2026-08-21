-- Initial storage schema for scraped League of Legends matchup chunks.
-- Embeddings are generated with 768 dimensions and compared with cosine distance.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS matchup_chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    champion TEXT NOT NULL,
    opponent TEXT NOT NULL,
    role TEXT NOT NULL CHECK (
        role IN ('top', 'jungle', 'mid', 'adc', 'support')
    ),

    source_url TEXT NOT NULL,
    tip TEXT NOT NULL DEFAULT '',
    stats JSONB NOT NULL DEFAULT '[]'::jsonb,
    chunk_text TEXT NOT NULL,

    -- Nullable so raw chunks can be ingested before embeddings are generated.
    embedding vector(768),

    -- The ingestion pipeline computes this from the content-bearing fields.
    content_hash TEXT NOT NULL,
    scraped_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT matchup_chunks_content_version_key
        UNIQUE (champion, opponent, role, source_url, content_hash)
);

CREATE INDEX IF NOT EXISTS matchup_chunks_champion_idx
    ON matchup_chunks (champion);

CREATE INDEX IF NOT EXISTS matchup_chunks_opponent_idx
    ON matchup_chunks (opponent);

CREATE INDEX IF NOT EXISTS matchup_chunks_role_idx
    ON matchup_chunks (role);

CREATE INDEX IF NOT EXISTS matchup_chunks_pair_role_idx
    ON matchup_chunks (champion, opponent, role);

-- Add an HNSW vector index after exact-search retrieval has an evaluated baseline.
