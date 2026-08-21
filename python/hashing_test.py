import json
from datetime import timedelta, timezone

import pytest

from ingest_chunks import compute_content_hash, load_chunks, parse_timestamp


def make_chunk() -> dict:
    return {
        "champion": "Malphite",
        "opponent": "Yone",
        "role": "top",
        "source_url": "https://op.gg/example",
        "tip": "Use E to reduce Yone's attack speed.",
        "stats": [
            {
                "label": "Win rate",
                "champion_value": "52.50%",
                "opponent_value": "47.50%",
            }
        ],
        "chunk_text": "Malphite vs Yone (Top lane matchup)",
        "created_at": "2026-08-19T03:55:31Z",
    }


def write_json(tmp_path, value):
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_load_chunks_returns_valid_chunks(tmp_path):
    chunk = make_chunk()

    assert load_chunks(write_json(tmp_path, [chunk])) == [chunk]


def test_load_chunks_rejects_non_list_document(tmp_path):
    path = write_json(tmp_path, make_chunk())

    with pytest.raises(ValueError, match="Expected the input JSON to contain a list"):
        load_chunks(path)


def test_load_chunks_rejects_non_object_chunk(tmp_path):
    path = write_json(tmp_path, ["not a chunk"])

    with pytest.raises(ValueError, match="Chunk 0 is not a JSON object"):
        load_chunks(path)


def test_load_chunks_reports_missing_fields_in_sorted_order(tmp_path):
    chunk = make_chunk()
    del chunk["champion"]
    del chunk["chunk_text"]

    with pytest.raises(
        ValueError,
        match="Chunk 0 is missing fields: champion, chunk_text",
    ):
        load_chunks(write_json(tmp_path, [chunk]))


def test_same_content_produces_same_hash():
    first = make_chunk()
    second = dict(reversed(list(first.items())))

    assert compute_content_hash(first) == compute_content_hash(second)


def test_scrape_timestamp_does_not_change_content_hash():
    first = make_chunk()
    second = make_chunk()
    second["created_at"] = "2026-08-20T03:55:31Z"

    assert compute_content_hash(first) == compute_content_hash(second)


@pytest.mark.parametrize("field", ["tip", "stats", "chunk_text"])
def test_content_change_produces_different_hash(field):
    original = make_chunk()
    changed = make_chunk()
    changed[field] = "different content"

    assert compute_content_hash(original) != compute_content_hash(changed)


def test_parse_timestamp_converts_z_to_utc():
    parsed = parse_timestamp("2026-08-19T03:55:31Z")

    assert parsed.tzinfo == timezone.utc
    assert parsed.utcoffset() == timedelta(0)


def test_parse_timestamp_preserves_explicit_offset():
    parsed = parse_timestamp("2026-08-18T20:55:31-07:00")

    assert parsed.utcoffset() == timedelta(hours=-7)


def test_parse_timestamp_rejects_invalid_value():
    with pytest.raises(ValueError):
        parse_timestamp("not-a-timestamp")
