"""RAG API. /v1/query is the Phase 2 hybrid+rerank+grounded endpoint (default).
/v1/query/naive is kept as the Phase 1 dense-only endpoint for comparison.

Run:
    uvicorn src.api.main:app --reload
"""
import logging
import os
import time

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.api.rate_limit import enforce_rate_limit
from src.generation.grounded_client import generate_grounded_answer
from src.generation.llm_client import generate_answer
from src.observability.logger import Timer, log_query
from src.rerank.cross_encoder import _get_model
from src.retrieval.dense import retrieve
from src.retrieval.hybrid import retrieve_hybrid

app = FastAPI(title="RAG Production - Phase 2")

# Fail-closed by default: no origins allowed until CORS_ALLOWED_ORIGINS is set.
# This only matters for browser-based callers; server-to-server clients
# (curl, backend services) aren't subject to CORS at all.
_allowed_origins = [o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.middleware("http")
async def log_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    client_host = request.client.host if request.client else "client"
    print(
        f"INFO:     {client_host} - \"{request.method} {request.url.path}\" "
        f"{response.status_code} - Completed in {duration:.2f}s"
    )
    response.headers["X-Process-Time"] = f"{duration:.2f}s"
    return response


@app.on_event("startup")
def warmup_models():
    """Pre-load the cross-encoder reranker model into memory on server boot."""
    _get_model()


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


@app.post(
    "/v1/query", response_model=GroundedQueryResponse, dependencies=[Depends(enforce_rate_limit)]
)
def query(request: QueryRequest):
    """Phase 2: dense + BM25 -> RRF fusion -> cross-encoder rerank -> grounded generation."""
    with Timer() as t:
        debug = retrieve_hybrid(request.query)
        result = generate_grounded_answer(request.query, debug.reranked_results)

    log_query(request.query, debug, result, latency_seconds=t.elapsed, timestamp=t.start)

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


@app.post(
    "/v1/query/naive", response_model=QueryResponse, dependencies=[Depends(enforce_rate_limit)]
)
def query_naive(request: QueryRequest):
    """Phase 1: dense-only retrieval, no citation validation. Kept for comparison."""
    chunks = retrieve(request.query, top_k=request.top_k)
    answer = generate_answer(request.query, chunks)
    return QueryResponse(answer=answer, retrieved_chunk_ids=[c.chunk_id for c in chunks])
