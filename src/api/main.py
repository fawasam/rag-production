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

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.api.rate_limit import enforce_rate_limit
from src.generation.grounded_client import generate_grounded_answer
from src.generation.llm_client import generate_answer
from src.ingestion.index import build_index
from src.ingestion.parsers import SUPPORTED_EXTENSIONS
from src.ingestion.watcher import start_watcher
from src.observability.logger import LOG_PATH, Timer, log_query
from src.rerank.cross_encoder import _get_model
from src.retrieval.dense import retrieve
from src.retrieval.hybrid import retrieve_hybrid
from src.security.guardrails import check_prompt_injection
from src.security.pii_masker import mask_pii

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
    """Pre-load cross-encoder model & start background auto-watcher on boot."""
    _get_model()
    try:
        start_watcher()
    except Exception as e:
        print(f"WARNING: Could not start auto-watcher: {e}")



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
    """Phase 2: Security checks -> Dense + BM25 -> RRF fusion -> Rerank -> Grounded generation."""
    # 1. Prompt Injection Shield
    is_safe, vtype, reason = check_prompt_injection(request.query)
    if not is_safe:
        security_info = {
            "injection_blocked": True,
            "violation_type": vtype,
            "reason": reason,
            "pii_detected": 0,
            "pii_entities": [],
            "masked_query": request.query,
        }
        log_query(
            request.query, None, None, latency_seconds=0.01, timestamp=time.time(), security_debug=security_info
        )
        return GroundedQueryResponse(
            answer=f"🛡️ Security Refusal: Request blocked due to prompt injection detection ({reason}).",
            answerable=False,
            citations_valid=False,
            citations=[],
            retrieved_chunk_ids=[],
            debug={"security": security_info},
        )

    # 2. PII Redaction Engine
    masked_query, pii_entities = mask_pii(request.query)
    security_info = {
        "injection_blocked": False,
        "pii_detected": len(pii_entities),
        "pii_entities": pii_entities,
        "masked_query": masked_query,
    }

    # 3. Retrieval & Grounded Generation using Masked Query
    with Timer() as t:
        debug = retrieve_hybrid(masked_query)
        result = generate_grounded_answer(masked_query, debug.reranked_results)

    log_query(
        request.query, debug, result, latency_seconds=t.elapsed, timestamp=t.start, security_debug=security_info
    )

    return GroundedQueryResponse(
        answer=result.answer,
        answerable=result.answerable,
        citations_valid=result.citations_valid,
        citations=result.citations,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        debug={
            "security": security_info,
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


@app.post("/v1/ingest", dependencies=[Depends(enforce_rate_limit)])
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload a raw document (.pdf, .docx, .md, .txt) and trigger background delta re-indexing."""
    filename = file.filename or "uploaded_doc"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    target_path = RAW_DATA_PATH / filename

    contents = await file.read()
    with open(target_path, "wb") as f:
        f.write(contents)

    background_tasks.add_task(build_index, force_full=False)

    return {
        "status": "uploaded",
        "filename": filename,
        "size_bytes": len(contents),
        "message": f"File '{filename}' uploaded successfully. Background re-indexing scheduled."
    }


@app.get("/v1/documents", dependencies=[Depends(enforce_rate_limit)])
def list_documents():
    """List all existing raw documents in data/raw/ with chunk count from ingestion manifest."""
    manifest = {}
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as mf:
                manifest = json.load(mf)
        except Exception:
            pass

    docs = []
    if RAW_DATA_PATH.exists():
        files = sorted(
            [f for f in RAW_DATA_PATH.iterdir() if f.is_file() and not f.name.startswith(".")],
            key=lambda x: x.name.lower()
        )
        for f in files:
            ext = f.suffix.lower().replace(".", "")
            manifest_info = manifest.get(f.name, {})
            chunk_count = len(manifest_info.get("chunk_ids", []))
            docs.append({
                "filename": f.name,
                "format": ext.upper() if ext else "OTHER",
                "size_bytes": f.stat().st_size,
                "mtime": f.stat().st_mtime,
                "chunk_count": chunk_count
            })

    return {"total": len(docs), "documents": docs}


@app.delete("/v1/documents/{filename}", dependencies=[Depends(enforce_rate_limit)])
def delete_document(filename: str, background_tasks: BackgroundTasks):
    """Delete a raw document from data/raw/ and trigger background index purge."""
    target_path = RAW_DATA_PATH / filename
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found."
        )

    target_path.unlink()
    background_tasks.add_task(build_index, force_full=False)

    return {"status": "deleted", "filename": filename, "message": f"Document '{filename}' deleted. Index updated."}




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


