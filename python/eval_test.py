import json

import pytest

from eval import load_dataset, rank_expected_source, score_answer, summarize


def test_load_dataset_accepts_valid_cases(tmp_path):
    path = tmp_path / "cases.json"
    case = {
        "id": "one", "question": "q", "champion": "Malphite",
        "opponent": "Yone", "role": "top", "expected_source_url": "url",
        "expected_terms": ["armor"],
    }
    path.write_text(json.dumps([case]), encoding="utf-8")
    assert load_dataset(path) == [case]


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "cases.json"
    case = {
        "id": "same", "question": "q", "champion": "Malphite",
        "opponent": "Yone", "role": "top", "expected_source_url": "url",
        "expected_terms": ["armor"],
    }
    path.write_text(json.dumps([case, case]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate evaluation id"):
        load_dataset(path)


def test_rank_expected_source_uses_one_based_rank():
    matches = [{"source_url": "wrong"}, {"source_url": "expected"}]
    assert rank_expected_source(matches, "expected") == 2
    assert rank_expected_source(matches, "missing") is None


def test_score_answer_measures_terms_and_valid_citations():
    score = score_answer("Use E for ATTACK SPEED. [Source 1]", ["E", "attack speed", "W"], 1)
    assert score["expected_term_recall"] == pytest.approx(2 / 3)
    assert score["matched_terms"] == ["E", "attack speed"]
    assert score["has_valid_citation"] is True


def test_score_answer_does_not_match_single_letter_inside_words():
    score = score_answer("Use armor carefully. [Source 1]", ["E"], 1)
    assert score["expected_term_recall"] == 0.0


def test_summarize_calculates_retrieval_and_answer_metrics():
    results = [
        {"expected_source_rank": 1, "retrieved_sources": ["a"],
         "answer_scores": {"expected_term_recall": 1.0, "has_valid_citation": True}},
        {"expected_source_rank": None, "retrieved_sources": [],
         "answer_scores": {"expected_term_recall": 0.0, "has_valid_citation": False}},
    ]
    metrics = summarize(results, 3)
    assert metrics["hit_at_3"] == 0.5
    assert metrics["mrr"] == 0.5
    assert metrics["no_result_rate"] == 0.5
    assert metrics["mean_expected_term_recall"] == 0.5
    assert metrics["citation_rate"] == 0.5
