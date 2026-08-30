"""Phase 1 generation: stuff retrieved chunks into the prompt, ask GPT to answer.

No citation-schema enforcement yet — that's a Phase 2 requirement (SRS FR-2.5/FR-2.6).
This is intentionally the "naive" version: it prints sources, but doesn't
mechanically verify every claim maps to one.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval.dense import RetrievedChunk

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a support assistant. Answer the user's question using ONLY the
provided context chunks. If the context does not contain the answer, say you don't know —
do not make anything up. After your answer, list the chunk_ids you used under "Sources:"."""


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
    return OpenAI(api_key=api_key)


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[chunk_id: {c.chunk_id}]\n{c.text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> str:
    client = get_openai_client()
    context_block = build_context_block(chunks)

    user_prompt = f"""Context:
{context_block}

Question: {query}"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content
