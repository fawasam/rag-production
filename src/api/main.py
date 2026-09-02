"""RAG API. /v1/query is the Phase 2 hybrid+rerank+grounded endpoint (default).
/v1/query/naive is kept as the Phase 1 dense-only endpoint for comparison.

Run:
    uvicorn src.api.main:app --reload
"""
import json
import logging
import os
import time

from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.rate_limit import enforce_rate_limit
from src.generation.grounded_client import generate_grounded_answer
from src.generation.llm_client import generate_answer
from src.observability.logger import LOG_PATH, Timer, log_query
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


RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw"
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "ingestion_manifest.json"


@app.get("/v1/logs", dependencies=[Depends(enforce_rate_limit)])
def get_query_logs(limit: int = 50):
    """Retrieve real query execution telemetry, document inventory, format breakdown, and trends."""
    total_docs = 0
    file_formats = {}
    if RAW_DATA_PATH.exists():
        files = [f for f in RAW_DATA_PATH.iterdir() if f.is_file() and not f.name.startswith(".")]
        total_docs = len(files)
        for f in files:
            ext = f.suffix.lower().replace(".", "")
            label = ext.upper() if ext else "OTHER"
            file_formats[label] = file_formats.get(label, 0) + 1

    total_chunks = 0
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as mf:
                manifest_data = json.load(mf)
                total_chunks = sum(len(entry.get("chunk_ids", [])) for entry in manifest_data.values())
        except Exception:
            pass

    if not LOG_PATH.exists():
        return {
            "summary": {
                "total_queries": 0,
                "avg_latency_seconds": 0.0,
                "citation_valid_rate": 100.0,
                "answerable_rate": 100.0,
                "total_docs": total_docs,
                "total_chunks": total_chunks,
                "file_formats": file_formats,
            },
            "logs": [],
        }

    logs = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total_queries = len(logs)
    if total_queries == 0:
        return {
            "summary": {
                "total_queries": 0,
                "avg_latency_seconds": 0.0,
                "citation_valid_rate": 100.0,
                "answerable_rate": 100.0,
                "total_docs": total_docs,
                "total_chunks": total_chunks,
                "file_formats": file_formats,
            },
            "logs": [],
        }

    avg_latency = round(sum(l.get("latency_seconds", 0) for l in logs) / total_queries, 2)
    valid_citations_count = sum(1 for l in logs if l.get("citations_valid", False))
    answerable_count = sum(1 for l in logs if l.get("answerable", False))

    citation_valid_rate = round((valid_citations_count / total_queries) * 100, 1)
    answerable_rate = round((answerable_count / total_queries) * 100, 1)

    recent_logs = logs[-limit:][::-1]

    return {
        "summary": {
            "total_queries": total_queries,
            "avg_latency_seconds": avg_latency,
            "citation_valid_rate": citation_valid_rate,
            "answerable_rate": answerable_rate,
            "total_docs": total_docs,
            "total_chunks": total_chunks,
            "file_formats": file_formats,
        },
        "logs": recent_logs,
    }




static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    from fastapi.responses import FileResponse

    @app.get("/", include_in_schema=False)
    def read_root():
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


