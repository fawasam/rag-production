"""Phase 1 API: single /v1/query endpoint (see SRS.md section 4.2).

Run:
    uvicorn src.api.main:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.generation.llm_client import generate_answer
from src.retrieval.dense import retrieve

app = FastAPI(title="RAG Production - Phase 1")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunk_ids: list[str]


@app.get("/v1/health")
def health():
    return {"status": "ok"}


@app.post("/v1/query", response_model=QueryResponse)
def query(request: QueryRequest):
    chunks = retrieve(request.query, top_k=request.top_k)
    answer = generate_answer(request.query, chunks)
    return QueryResponse(answer=answer, retrieved_chunk_ids=[c.chunk_id for c in chunks])
