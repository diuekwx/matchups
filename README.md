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
