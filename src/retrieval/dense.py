"""Phase 1 retrieval: plain dense similarity search over the Chroma index."""
from dataclasses import dataclass

import chromadb

from src.ingestion.index import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, get_openai_client


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    doc_id: str
    source_path: str
    score: float  # similarity score, higher = more relevant


def _get_collection():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma_client.get_collection(COLLECTION_NAME)


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    client = get_openai_client()
    query_embedding = (
        client.embeddings.create(model=EMBEDDING_MODEL, input=[query]).data[0].embedding
    )

    collection = _get_collection()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]  # cosine distance, lower = more similar

    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                doc_id=meta["doc_id"],
                source_path=meta["source_path"],
                score=1 - distance,  # convert to a similarity-style score
            )
        )
    return retrieved


if __name__ == "__main__":
    for r in retrieve("Does the Team plan support SSO?", top_k=3):
        print(f"[{r.score:.3f}] {r.chunk_id}\n{r.text[:200]}...\n")
