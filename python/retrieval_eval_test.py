import hashlib

from retrieval_eval import (
    TfidfIndex,
    build_passages,
    infer_stat_intent,
    file_sha256,
    paired_comparison,
    passage_id,
    retrieve_hybrid,
    reproducibility_metadata,
    summarize_latency,
    summarize,
)


def test_file_sha256_hashes_exact_file_bytes(tmp_path):
    path = tmp_path / "input.json"
    content = b'{"stable":true}\n'
    path.write_bytes(content)

    assert file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_reproducibility_metadata_identifies_inputs_code_and_config(tmp_path):
    corpus_path = tmp_path / "chunks.json"
    dataset_path = tmp_path / "cases.json"
    evaluator_path = tmp_path / "project" / "python" / "retrieval_eval.py"
    evaluator_path.parent.mkdir(parents=True)
    corpus_path.write_text("[]", encoding="utf-8")
    dataset_path.write_text("[]", encoding="utf-8")
    evaluator_path.write_text("# evaluator", encoding="utf-8")

    metadata = reproducibility_metadata(
        corpus_path,
        dataset_path,
        evaluator_path,
        chunks=[{"id": 1}],
        passages=[{"id": 1}, {"id": 2}],
        cases=[{"id": "case-1"}],
        latency_runs=20,
    )

    assert metadata["report_schema_version"] == "2.1"
    assert metadata["inputs"]["corpus"]["chunks"] == 1
    assert metadata["inputs"]["corpus"]["passages"] == 2
    assert metadata["inputs"]["dataset"]["cases"] == 1
    assert len(metadata["inputs"]["corpus"]["sha256"]) == 64
    assert len(metadata["code"]["evaluator_sha256"]) == 64
    assert metadata["retrieval_config"]["limit"] == 10
    assert metadata["retrieval_config"]["latency_runs_per_case"] == 20
    assert metadata["retrieval_config"]["hybrid"]["metadata_filters"] == [
        "champion", "opponent", "role",
    ]
    assert metadata["models"]["embedding"] is None


def test_build_passages_splits_tip_and_stats():
    chunks = [{
        "champion": "Ahri", "opponent": "Zed", "role": "mid", "source_url": "u",
        "tip": "Hold Charm.",
        "stats": [{"label": "Win rate", "champion_value": "51%", "opponent_value": "49%"}],
    }]
    passages = build_passages(chunks)
    assert [p["id"] for p in passages] == [passage_id("u", "tip"), passage_id("u", "stat:win_rate")]


def test_infer_stat_intent_handles_paraphrases_and_specificity():
    assert infer_stat_intent("What is the overall success rate?") == "win_rate"
    assert infer_stat_intent("Who wins the laning phase more often?") == "lane_win_rate"
    assert infer_stat_intent("How should I play this matchup?") is None


def test_hybrid_filters_pair_and_ranks_inferred_stat_first():
    passages = [
        {"id": "a-tip", "champion": "Ahri", "opponent": "Zed", "role": "mid", "source_url": "a", "evidence_type": "tip", "text": "Ahri Zed tip"},
        {"id": "a-win", "champion": "Ahri", "opponent": "Zed", "role": "mid", "source_url": "a", "evidence_type": "stat:win_rate", "text": "Ahri Zed win rate 51 49"},
        {"id": "b-win", "champion": "Yasuo", "opponent": "Zed", "role": "mid", "source_url": "b", "evidence_type": "stat:win_rate", "text": "Yasuo Zed win rate 55 45"},
    ]
    results = retrieve_hybrid(TfidfIndex(passages), {
        "question": "What is their overall success rate?", "champion": "Ahri", "opponent": "Zed", "role": "mid",
    }, 3)
    assert [result["id"] for result in results] == ["a-win", "a-tip"]


def test_summarize_counts_misses_and_ranks():
    metrics = summarize([{"rank": 1}, {"rank": 3}, {"rank": None}])
    assert metrics["hit_at_1"] == 1 / 3
    assert metrics["hit_at_3"] == 2 / 3
    assert metrics["mrr_at_10"] == (1 + 1 / 3) / 3
    assert metrics["miss_at_10"] == 1 / 3


def test_summarize_latency_reports_distribution_and_total():
    latency = summarize_latency([4.0, 1.0, 3.0, 2.0])

    assert latency == {
        "queries": 4,
        "min_ms": 1.0,
        "mean_ms": 2.5,
        "median_ms": 2.5,
        "p95_ms": 4.0,
        "max_ms": 4.0,
        "estimated_workload_total_ms": 10.0,
    }


def test_paired_comparison_counts_rank_and_top_one_changes():
    baseline = {"results": [{"id": "a", "rank": None}, {"id": "b", "rank": 1}, {"id": "c", "rank": 3}]}
    hybrid = {"results": [{"id": "a", "rank": 1}, {"id": "b", "rank": 2}, {"id": "c", "rank": 1}]}
    comparison = paired_comparison(baseline, hybrid)
    assert comparison["rank_wins"] == 2
    assert comparison["rank_losses"] == 1
    assert comparison["hit_at_1_improvements"] == 2
    assert comparison["hit_at_1_regressions"] == 1
