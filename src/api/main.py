"""RAG API. /v1/query is the Phase 2 hybrid+rerank+grounded endpoint (default).
/v1/query/naive is kept as the Phase 1 dense-only endpoint for comparison.

Run:
    uvicorn src.api.main:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.generation.grounded_client import generate_grounded_answer
from src.generation.llm_client import generate_answer
from src.retrieval.dense import retrieve
from src.retrieval.hybrid import retrieve_hybrid

app = FastAPI(title="RAG Production - Phase 2")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunk_ids: list[str]


class GroundedQueryResponse(BaseModel):
    answer: str
    answerable: bool
    citations_valid: bool
    citations: list[dict]
    retrieved_chunk_ids: list[str]
    debug: dict


@app.get("/v1/health")
def health():
    return {"status": "ok"}


@app.post("/v1/query", response_model=GroundedQueryResponse)
def query(request: QueryRequest):
    """Phase 2: dense + BM25 -> RRF fusion -> cross-encoder rerank -> grounded generation."""
    debug = retrieve_hybrid(request.query)
    result = generate_grounded_answer(request.query, debug.reranked_results)

    return GroundedQueryResponse(
        answer=result.answer,
        answerable=result.answerable,
        citations_valid=result.citations_valid,
        citations=result.citations,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        debug={
            "dense": [c.chunk_id for c in debug.dense_results],
            "bm25": [c.chunk_id for c in debug.bm25_results],
            "fused": [c.chunk_id for c in debug.fused_results],
            "reranked": [c.chunk_id for c in debug.reranked_results],
            "invalid_citations": [vars(ic) for ic in result.invalid_citations],
        },
    )


@app.post("/v1/query/naive", response_model=QueryResponse)
def query_naive(request: QueryRequest):
    """Phase 1: dense-only retrieval, no citation validation. Kept for comparison."""
    chunks = retrieve(request.query, top_k=request.top_k)
    answer = generate_answer(request.query, chunks)
    return QueryResponse(answer=answer, retrieved_chunk_ids=[c.chunk_id for c in chunks])
