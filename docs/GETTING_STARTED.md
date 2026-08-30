# Getting Started — Production-Grade RAG

This is the build order to go from an empty repo to Phase 3, matching [SRS.md](./SRS.md).

## 0. Repo scaffolding (Day 1)

```bash
cd /Users/fawasam/Documents/work/AI/rag-production
git init
mkdir -p src/{ingestion,retrieval,rerank,generation,eval,api} data/raw data/processed eval
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn chromadb rank_bm25 sentence-transformers anthropic ragas pytest
```

Suggested skeleton:
```
rag-production/
├── docs/
│   ├── SRS.md
│   └── GETTING_STARTED.md
├── src/
│   ├── ingestion/      # parse, chunk, embed, index
│   ├── retrieval/      # bm25.py, dense.py, rrf.py
│   ├── rerank/          # cross_encoder.py
│   ├── generation/     # prompt_builder.py, llm_client.py
│   ├── eval/            # ragas/trulens harness
│   └── api/             # FastAPI app (main.py)
├── data/
│   ├── raw/              # source docs
│   └── processed/        # chunked + embedded output
├── eval/
│   └── golden_set.jsonl  # curated Q&A for CI gating
└── tests/
```

## 1. Phase 1 — Naive RAG (aim: 3-5 days)
1. Drop 10-20 sample documents into `data/raw/`.
2. Write `ingestion/chunk.py` — fixed-size chunking (e.g. 500 tokens, 50 overlap).
3. Write `ingestion/embed.py` — embed chunks (start with a free/local model via `sentence-transformers` to avoid API cost while prototyping).
4. Stand up Chroma locally, write `ingestion/index.py` to load chunks + embeddings.
5. Write `retrieval/dense.py` — embed the query, top-k similarity search.
6. Write `generation/llm_client.py` — pass top-k chunks + query to an LLM (Claude), return the answer.
7. Wire it behind a single `/v1/query` FastAPI endpoint. **Milestone: ask a question, get an answer.**

## 2. Phase 2 — Production-Grade (aim: 2-3 weeks)
1. Add `retrieval/bm25.py` (start with `rank_bm25` in-memory; migrate to OpenSearch later if the corpus grows).
2. Add `retrieval/rrf.py` implementing Reciprocal Rank Fusion to merge BM25 + dense ranked lists.
3. Add `rerank/cross_encoder.py` using a pretrained cross-encoder (e.g. `ms-marco-MiniLM-L-6-v2`) to re-score the fused top-N (start N=20 → M=5).
4. Update `generation/prompt_builder.py` to enforce a strict JSON schema: every claim must reference a `chunk_id`. Validate the LLM's JSON output server-side — reject/retry if a citation doesn't map to a real retrieved chunk.
5. Curate `eval/golden_set.jsonl`: 20-50 (question, expected_answer, expected_chunk_ids) rows from your actual domain.
6. Add `eval/run_ragas.py` — score faithfulness, context precision/recall, answer relevance against the golden set.
7. Add a GitHub Actions workflow (`.github/workflows/eval.yml`) that runs the eval suite on every PR and fails the build below the thresholds in SRS §8.

## 3. Phase 3 — Observability (aim: ongoing)
1. Log every production query: retrieved chunk IDs, scores, final prompt, answer, citations, latency.
2. Add a scheduled job (or TruLens dashboard) that samples production traffic and re-scores faithfulness/precision over time.
3. Set up an alert (Slack/PagerDuty) when rolling faithfulness or citation-validity drops below threshold.
4. Before swapping embedding models, chunking strategy, or the re-ranker, run a shadow/A-B eval against the golden set first.

## First concrete command to run today

```bash
mkdir -p src/{ingestion,retrieval,rerank,generation,eval,api} data/raw data/processed eval tests
python3 -m venv .venv && source .venv/bin/activate && pip install fastapi uvicorn chromadb rank_bm25 sentence-transformers anthropic ragas pytest
```
Then drop a handful of real documents into `data/raw/` and start with `ingestion/chunk.py`.
