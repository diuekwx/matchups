"""Deterministic, API-free retrieval evaluation over the scraped matchup corpus.

The corpus is split into evidence-level passages (one tip or one statistic) so
retrieval can fail even after finding the correct matchup page.  Two strategies
are intentionally kept small and explainable:

* baseline: global TF-IDF cosine search over every passage
* hybrid: champion/opponent/role filtering plus lightweight intent expansion
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CORPUS = Path("scraper/output/chunks.json")
DEFAULT_DATASET = Path("eval/retrieval_cases.json")
DEFAULT_OUT = Path("eval/reports/retrieval_comparison.json")

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
STAT_ALIASES = {
    "win_rate": ("overall win rate", "wins overall", "overall success rate", "match win percentage"),
    "lane_win_rate": ("lane win rate", "wins lane", "wins the laning phase", "laning success rate", "lane advantage percentage"),
    "lane_kill_rate": ("lane kill rate", "solo kill rate", "kills in lane", "lane kills percentage"),
    "kda": ("kda", "kills deaths assists", "combat ratio"),
    "kill_participation": ("kill participation", "team kills involved", "kp"),
    "damage_dealt_to_champions": ("champion damage", "damage to champions", "player damage"),
    "first_tower_kill": ("first tower", "first turret", "tower time", "turret timing"),
    "lane_pick_rate": ("lane pick rate", "picked in lane", "lane popularity"),
    "ban_rate": ("ban rate", "banned", "ban percentage"),
}


def slug(value: str) -> str:
    return "_".join(TOKEN_RE.findall(value.casefold()))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def passage_id(source_url: str, evidence_type: str) -> str:
    return f"{source_url}#{evidence_type}"


def build_passages(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    for chunk in chunks:
        common = {
            "champion": chunk["champion"],
            "opponent": chunk["opponent"],
            "role": chunk["role"],
            "source_url": chunk["source_url"],
        }
        header = f'{chunk["champion"]} versus {chunk["opponent"]} {chunk["role"]}'
        if chunk.get("tip", "").strip():
            passages.append({
                **common,
                "id": passage_id(chunk["source_url"], "tip"),
                "evidence_type": "tip",
                "text": f'{header}. Matchup tip: {chunk["tip"]}',
            })
        for stat in chunk.get("stats", []):
            stat_type = f'stat:{slug(stat["label"])}'
            passages.append({
                **common,
                "id": passage_id(chunk["source_url"], stat_type),
                "evidence_type": stat_type,
                "text": (
                    f'{header}. {stat["label"]}: {chunk["champion"]} '
                    f'{stat["champion_value"]}; {chunk["opponent"]} '
                    f'{stat["opponent_value"]}.'
                ),
            })
    return passages


class TfidfIndex:
    def __init__(self, passages: list[dict[str, Any]]):
        self.passages = passages
        self.term_counts = [Counter(tokenize(p["text"])) for p in passages]
        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        count = len(passages)
        self.idf = {
            term: math.log((count + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }
        self.doc_norms = [self._norm(counts) for counts in self.term_counts]

    def _norm(self, counts: Counter[str]) -> float:
        return math.sqrt(sum((frequency * self.idf.get(term, 0.0)) ** 2 for term, frequency in counts.items()))

    def search(
        self,
        query: str,
        candidate_indices: Iterable[int] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        query_counts = Counter(tokenize(query))
        query_norm = self._norm(query_counts)
        indices = range(len(self.passages)) if candidate_indices is None else candidate_indices
        scored = []
        for index in indices:
            dot = sum(
                frequency * self.term_counts[index].get(term, 0) * self.idf.get(term, 0.0) ** 2
                for term, frequency in query_counts.items()
            )
            denominator = query_norm * self.doc_norms[index]
            score = dot / denominator if denominator else 0.0
            scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [{**self.passages[index], "score": score} for score, index in scored[:limit]]


def infer_stat_intent(question: str) -> str | None:
    normalized = " ".join(tokenize(question))
    matches = []
    for stat_type, aliases in STAT_ALIASES.items():
        for alias in aliases:
            alias_tokens = tokenize(alias)
            if " ".join(alias_tokens) in normalized:
                matches.append((len(alias_tokens), stat_type))
    return max(matches, default=(0, None))[1]


def expanded_query(question: str, intent: str | None) -> str:
    if not intent:
        return f"{question} matchup tip advice strategy"
    canonical = intent.replace("_", " ")
    return f"{question} {canonical} statistic"


def retrieve_baseline(index: TfidfIndex, case: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    return index.search(case["question"], limit=limit)


def retrieve_hybrid(index: TfidfIndex, case: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    candidates = [
        i for i, passage in enumerate(index.passages)
        if passage["champion"].casefold() == case["champion"].casefold()
        and passage["opponent"].casefold() == case["opponent"].casefold()
        and passage["role"].casefold() == case["role"].casefold()
    ]
    intent = infer_stat_intent(case["question"])
    results = index.search(expanded_query(case["question"], intent), candidates, limit=max(limit, len(candidates)))
    expected_type = f"stat:{intent}" if intent else "tip"
    results.sort(key=lambda item: (item["evidence_type"] != expected_type, -item["score"], item["id"]))
    return results[:limit]


def load_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    return value


def validate_cases(cases: list[dict[str, Any]], passages_by_id: dict[str, dict[str, Any]]) -> None:
    required = {"id", "question", "champion", "opponent", "role", "expected_passage_id", "challenge"}
    seen = set()
    for case in cases:
        missing = required - case.keys()
        if missing:
            raise ValueError(f'{case.get("id", "<unknown>")} missing fields: {sorted(missing)}')
        if case["id"] in seen:
            raise ValueError(f'Duplicate case id: {case["id"]}')
        seen.add(case["id"])
        if case["expected_passage_id"] not in passages_by_id:
            raise ValueError(f'{case["id"]} references a passage absent from the corpus')
        expected = passages_by_id[case["expected_passage_id"]]
        metadata = (case["champion"], case["opponent"], case["role"])
        expected_metadata = (expected["champion"], expected["opponent"], expected["role"])
        if metadata != expected_metadata:
            raise ValueError(f'{case["id"]} metadata does not match its expected passage')


def rank_of(results: list[dict[str, Any]], expected_id: str) -> int | None:
    return next((rank for rank, result in enumerate(results, 1) if result["id"] == expected_id), None)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    ranks = [result["rank"] for result in results]
    return {
        "cases": count,
        "hit_at_1": sum(rank == 1 for rank in ranks) / count,
        "hit_at_3": sum(rank is not None and rank <= 3 for rank in ranks) / count,
        "mrr_at_10": sum(1 / rank if rank else 0 for rank in ranks) / count,
        "miss_at_10": sum(rank is None for rank in ranks) / count,
    }


def evaluate(index: TfidfIndex, cases: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    retriever = retrieve_baseline if strategy == "baseline" else retrieve_hybrid
    results = []
    for case in cases:
        retrieved = retriever(index, case, 10)
        results.append({
            "id": case["id"],
            "challenge": case["challenge"],
            "rank": rank_of(retrieved, case["expected_passage_id"]),
            "expected_passage_id": case["expected_passage_id"],
            "top_passage_ids": [item["id"] for item in retrieved[:3]],
        })
    by_challenge = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["challenge"]].append(result)
    for challenge, challenge_results in sorted(grouped.items()):
        by_challenge[challenge] = summarize(challenge_results)
    return {"metrics": summarize(results), "by_challenge": by_challenge, "results": results}


def paired_comparison(baseline: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    baseline_by_id = {result["id"]: result for result in baseline["results"]}
    wins = losses = ties = 0
    top1_improvements = top1_regressions = 0
    for improved in hybrid["results"]:
        original = baseline_by_id[improved["id"]]
        original_rank = original["rank"] or math.inf
        improved_rank = improved["rank"] or math.inf
        if improved_rank < original_rank:
            wins += 1
        elif improved_rank > original_rank:
            losses += 1
        else:
            ties += 1
        top1_improvements += original["rank"] != 1 and improved["rank"] == 1
        top1_regressions += original["rank"] == 1 and improved["rank"] != 1

    discordant = top1_improvements + top1_regressions
    smaller = min(top1_improvements, top1_regressions)
    # Exact two-sided sign test on paired hit@1 outcomes (equivalent to exact
    # McNemar for discordant pairs), computed without scipy.
    p_value = min(
        1.0,
        2 * sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2 ** discordant),
    ) if discordant else 1.0
    return {
        "rank_wins": wins,
        "rank_losses": losses,
        "rank_ties": ties,
        "hit_at_1_improvements": top1_improvements,
        "hit_at_1_regressions": top1_regressions,
        "exact_p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and hybrid retrieval on scraped evidence")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    passages = build_passages(load_json_list(args.corpus))
    cases = load_json_list(args.dataset)
    passages_by_id = {passage["id"]: passage for passage in passages}
    validate_cases(cases, passages_by_id)
    index = TfidfIndex(passages)
    baseline = evaluate(index, cases, "baseline")
    hybrid = evaluate(index, cases, "hybrid")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": str(args.corpus),
        "dataset": str(args.dataset),
        "corpus_passages": len(passages),
        "evaluation_scope": (
            "Evidence retrieval only; champion, opponent, and role are supplied "
            "as structured inputs. Entity extraction and answer generation are not scored."
        ),
        "case_distribution": {
            "roles": dict(sorted(Counter(case["role"] for case in cases).items())),
            "challenges": dict(sorted(Counter(case["challenge"] for case in cases).items())),
        },
        "strategies": {
            "baseline": "global TF-IDF cosine",
            "hybrid": "champion/opponent/role filter + intent-aware TF-IDF reranking",
        },
        "baseline": baseline,
        "hybrid": hybrid,
        "absolute_improvement": {
            key: hybrid["metrics"][key] - baseline["metrics"][key]
            for key in ("hit_at_1", "hit_at_3", "mrr_at_10")
        },
        "paired_comparison": paired_comparison(baseline, hybrid),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "corpus_passages": len(passages),
        "baseline": baseline["metrics"],
        "hybrid": hybrid["metrics"],
        "absolute_improvement": report["absolute_improvement"],
    }, indent=2))
    print(f"Saved report to {args.out}")


if __name__ == "__main__":
    main()
