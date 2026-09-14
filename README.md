# League of Legends Matchup RAG

## Overview

A retrieval-augmented generation system that answers League of Legends matchup questions using scraped OP.GG tips and statistics. For example: "How should Malphite handle Jayce's ranged pressure in top lane?"

The project combines a concurrent Go scraper, PostgreSQL storage, metadata-filtered vector search, and Gemini answer generation. Answers are prompted to use retrieved evidence and cite their sources. An evaluation harness tracks retrieval performance, expected-term recall, and citation presence across pipeline changes.

The core pipeline is implemented directly in Go and Python, without LangChain or LangGraph.

## Requirements

- Go 1.26 or newer for the scraper.
- Python with pip for ingestion, embeddings, retrieval, and evaluation.
- Docker with Docker Compose for local PostgreSQL and pgvector.
- A Gemini API key for embeddings and answer generation.

After downloading or cloning the repository, install dependencies from the project root:

```powershell
python -m pip install -r python/requirements.txt
cd scraper
go mod download
cd ..
```

Create a `.env` file in the project root:

```dotenv
DATABASE_URL=postgresql://matchups:matchups_dev@localhost:5433/matchups
GEMINI_API_KEY=your_api_key
```

Do not commit API keys. The database URL above matches the included local Docker Compose configuration. Gemini API calls may incur costs.

## Architecture

```mermaid
flowchart TD
    A[OP.GG matchup pages] --> B[Go scraper]
    B --> C[Matchup JSON chunks]
    C --> D[Python validation and deduplication]
    D --> E[(PostgreSQL + pgvector)]
    E --> F[Gemini document embeddings]
    F --> E
    G[Question] --> H[Gemini query embedding]
    H --> I[Metadata filters + cosine similarity search]
    J[Champion, opponent, and role] --> I
    E --> I
    I --> K[Retrieved context]
    K --> L[Gemini answer + source citations]
    I --> M[Evaluation harness]
    L --> M
    M --> N[JSON metrics reports]
```

- **Scraping:** Bounded Go worker pools fetch matchup pages with shared rate limiting and exponential backoff. Chunks preserve matchup metadata, tips, statistics, and source URLs.
- **Storage and embeddings:** Python validates records, skips duplicate content using SHA-256 hashes, and stores 768-dimensional Gemini embeddings in PostgreSQL.
- **Retrieval and generation:** Explicit champion, opponent, and role filters narrow the candidate set before cosine similarity ranking. Gemini receives the retrieved context; empty retrieval returns an insufficient-data response.
- **Evaluation:** A repeatable question set measures source hit@k, mean reciprocal rank, no-result rate, expected-term recall, and citation presence.

## Retrieval benchmark

The original 10-case smoke evaluation had one database row per exact
champion/opponent/role filter, so every query retrieved the only candidate and
scored 100%. The evidence-level benchmark in `eval/retrieval_cases.json` is a
harder, deterministic ablation over the real scraped corpus:

- 45 hand-written cases, balanced across all five roles (9 each)
- 2,071 candidate passages: each matchup tip and each individual statistic is
  independently retrievable
- adversarial slices for implicit entities, reversed matchup direction,
  same-ability collisions, near-duplicate tips, and paraphrased stat intents
- paired comparison of global TF-IDF against metadata-filtered, intent-aware
  TF-IDF, with hit@1, hit@3, MRR@10, miss@10, challenge slices, and an exact
  paired significance test

Run it without a database, API key, or paid model call:

```powershell
$env:PYTHONPATH = "python"
python python/retrieval_eval.py
```

The checked-in report is `eval/reports/retrieval_comparison.json`. This
benchmark isolates retrieval: champion, opponent, and role are supplied as
structured inputs. It deliberately does not claim to evaluate entity extraction
or generated-answer quality; the existing `python/eval.py` remains the
end-to-end generation evaluation.

Each report includes a `reproducibility` block containing SHA-256 identifiers
for the corpus, evaluation dataset, and evaluator source; the Git commit and
dirty-worktree state; Python/platform information; and the complete retrieval
configuration. Matching hashes mean two reports used byte-for-byte identical
inputs and evaluator code. A dirty Git state is still traceable through the
evaluator hash, but committed experiments are preferable for final comparisons.

Latest checked-in result:

| Strategy | Hit@1 | Hit@3 | MRR@10 | Miss@10 |
| --- | ---: | ---: | ---: | ---: |
| Global TF-IDF baseline | 17.8% | 33.3% | 0.256 | 57.8% |
| Metadata + intent hybrid | 93.3% | 93.3% | 0.942 | 0.0% |

That is a +75.6 percentage-point Hit@1 improvement. The hybrid improves the
rank on 37 of 45 cases, regresses on none, and changes 34 paired Hit@1 failures
to successes (two-sided exact paired p = 1.16e-10). Three deliberately difficult
stat paraphrases remain below rank 1, keeping the result useful for the next
iteration instead of hiding all residual errors.
