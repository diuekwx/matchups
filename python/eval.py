import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from google import genai
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from answer import DEFAULT_GENERATION_MODEL, generate_answer
from retrieve import embed_query, require_environment_variable, retrieve_matches


DEFAULT_DATASET = Path("eval/matchup_questions.json")
DEFAULT_REPORT = Path("eval/reports/baseline.json")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must be a non-empty JSON list")

    required = {
        "id", "question", "champion", "opponent", "role",
        "expected_source_url", "expected_terms",
    }
    seen_ids = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Evaluation case {index} must be an object")
        missing = sorted(required - case.keys())
        if missing:
            raise ValueError(f"Evaluation case {index} missing: {', '.join(missing)}")
        if case["id"] in seen_ids:
            raise ValueError(f"Duplicate evaluation id: {case['id']}")
        seen_ids.add(case["id"])
        if not isinstance(case["expected_terms"], list) or not case["expected_terms"]:
            raise ValueError(f"Evaluation case {case['id']} needs expected_terms")
    return cases


def rank_expected_source(matches: list[dict[str, Any]], expected_url: str) -> int | None:
    for rank, match in enumerate(matches, start=1):
        if match["source_url"] == expected_url:
            return rank
    return None


def score_answer(answer: str, expected_terms: list[str], source_count: int) -> dict[str, Any]:
    lowered = answer.casefold()
    matched_terms = [
        term for term in expected_terms
        if re.search(
            rf"(?<!\w){re.escape(term.casefold())}(?!\w)",
            lowered,
        )
    ]
    valid_citations = [
        index for index in range(1, source_count + 1)
        if f"[source {index}]" in lowered
    ]
    return {
        "expected_term_recall": len(matched_terms) / len(expected_terms),
        "matched_terms": matched_terms,
        "has_valid_citation": bool(valid_citations),
        "valid_citations": valid_citations,
    }


def summarize(results: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    count = len(results)
    ranks = [result["expected_source_rank"] for result in results]
    return {
        "cases": count,
        f"hit_at_{limit}": sum(rank is not None for rank in ranks) / count,
        "mrr": sum(1 / rank if rank is not None else 0 for rank in ranks) / count,
        "no_result_rate": sum(not result["retrieved_sources"] for result in results) / count,
        "mean_expected_term_recall": sum(
            result["answer_scores"]["expected_term_recall"] for result in results
        ) / count,
        "citation_rate": sum(
            result["answer_scores"]["has_valid_citation"] for result in results
        ) / count,
    }


def run_evaluation(
    cases: list[dict[str, Any]],
    connection: psycopg.Connection,
    client: genai.Client,
    limit: int,
    generation_model: str,
    request_delay: float = 0.0,
) -> list[dict[str, Any]]:
    results = []
    for index, case in enumerate(cases, start=1):
        embedding = embed_query(client, case["question"])
        matches = retrieve_matches(
            connection, embedding, case["champion"], case["opponent"],
            case["role"], limit,
        )
        rank = rank_expected_source(matches, case["expected_source_url"])
        answer = (
            generate_answer(client, case["question"], matches, generation_model)
            if matches else ""
        )
        scores = score_answer(answer, case["expected_terms"], len(matches))
        results.append({
            "id": case["id"],
            "question": case["question"],
            "expected_source_url": case["expected_source_url"],
            "expected_source_rank": rank,
            "retrieved_sources": [match["source_url"] for match in matches],
            "answer": answer,
            "answer_scores": scores,
        })
        print(f"[{index}/{len(cases)}] {case['id']}: rank={rank}, "
              f"term_recall={scores['expected_term_recall']:.2f}")
        if request_delay and index < len(cases):
            time.sleep(request_delay)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate matchup RAG retrieval and answers")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--request-delay", type=float, default=13.0,
        help="Seconds between cases; default respects Gemini's 5 RPM free tier",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    cases = load_dataset(args.dataset)
    client = genai.Client(api_key=require_environment_variable("GEMINI_API_KEY"))
    model = os.getenv("GENERATION_MODEL", DEFAULT_GENERATION_MODEL)

    with psycopg.connect(
        require_environment_variable("DATABASE_URL"), row_factory=dict_row
    ) as connection:
        register_vector(connection)
        results = run_evaluation(
            cases, connection, client, args.limit, model, args.request_delay
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "retrieval_limit": args.limit,
        "generation_model": model,
        "request_delay_seconds": args.request_delay,
        "metrics": summarize(results, args.limit),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved report to {args.out}")


if __name__ == "__main__":
    main()
