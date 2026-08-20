# League of Legends Matchup RAG — Project Brief

## Goal
Build a RAG system that answers League of Legends matchup questions (e.g. "how do I beat Yasuo top as Malphite") by retrieving from real scraped matchup/guide data, not from the LLM's own training knowledge. The point of this project is to demonstrate real retrieval-augmented generation engineering for AI/MLE job applications — not to ship a polished consumer product.

## Why this project (context for Claude Code)
This is a portfolio/resume project targeting AI engineer and ML engineer roles. The differentiator vs. a typical tutorial RAG project is:
1. A self-built scraping/ingestion pipeline (not a pre-packaged dataset)
2. Hybrid retrieval — structured metadata filtering + vector search, not just naive embed-and-retrieve
3. A **real eval harness** — this is the most important part. Most RAG demos skip evaluation; this one should track retrieval precision/recall and answer quality across pipeline changes.

Do not add LangChain/LangGraph for the core pipeline — build the embed → store → retrieve → prompt loop directly so every step is explainable in an interview. LangGraph could be reconsidered later only if we build a multi-index query router (e.g. routing between matchup data, patch notes, ability tags).

## Architecture

### 1. Scraper (Go)
- Target: League of Legends Wiki (leagueoflegends.fandom.com) matchup/counter sections as primary source. Possibly expand later to guide sites (check robots.txt/ToS first) and summoner school-style community threads.
- Use a worker pool (goroutines + channels) for concurrent fetching, bounded to a reasonable concurrency limit.
- Respect `robots.txt`, add rate limiting (`golang.org/x/time/rate`) and exponential backoff on failures.
- Output: write scraped, cleaned chunks directly into Postgres (see schema below), OR write to intermediate JSON files if that's simpler to start.
- Chunk by matchup pair (one chunk = one champion vs. one opponent's laning/matchup notes), not by arbitrary token count.

### 2. Storage (Postgres + pgvector via Supabase)
Table: `matchup_chunks`
- `id`
- `champion` (text)
- `opponent` (text)
- `role` (text, e.g. top/jungle/mid/adc/support)
- `source_url` (text)
- `chunk_text` (text)
- `embedding` (vector)
- `created_at`

Metadata columns (`champion`, `opponent`, `role`) should be filterable BEFORE vector search — this is the "hybrid retrieval" part. E.g. a query mentioning "Malphite vs Yasuo top" should filter to that pair first, then use vector search within/around that filtered set for nuance.

### 3. Embedding + Generation (Python)
- Embeddings: Gemini embedding API (matches existing stack from other projects), or Ollama-local (`nomic-embed-text`) as a cost-free alternative — decide based on dataset size once scraping is done.
- Generation: Gemini API, prompted to answer only using the retrieved chunks (avoid the model falling back on its own training knowledge about LoL, which defeats the point of the RAG).

### 4. Eval harness (critical — do not skip or treat as optional)
- Build a test set of Q&A pairs (e.g. 30-50 hand-written matchup questions with known-good answers/sources).
- Track retrieval metrics: precision/recall — did the retrieved chunks actually contain the source info needed to answer correctly?
- Track answer quality: LLM-as-judge scoring (Gemini grading its own or another model's output against expected answer) or manually graded scoring.
- Structure this so it can be re-run after any pipeline change (different chunking strategy, different embedding model, different retrieval method) and produce a before/after comparison. This before/after tracking is the key resume differentiator — the goal is being able to say "improved retrieval precision from X% to Y% by [change]."

## Explicitly out of scope for v1
- LangChain/LangGraph (see above)
- Fine-tuning (that's a separate, later project — see LoRA/QLoRA plan)
- Full production deployment/monitoring — nice-to-have stretch goal, not core to the resume story here

## Suggested build order
1. Go scraper — get raw matchup text flowing into Postgres for a small subset of champions first (validate the pipeline end to end before scaling up)
2. DB schema + embedding script (Python) — embed and store the initial subset
3. Basic retrieval + generation loop — confirm hybrid filter + vector search works end to end for a few manual queries
4. Eval harness — build the test set and scoring pipeline
5. Scale scraping to full champion roster, re-run evals, iterate on chunking/retrieval based on eval results