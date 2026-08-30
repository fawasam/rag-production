"""Phase 2 generation: schema-enforced answers with server-validated citations.

Implements FR-2.5 (structured citation schema) and FR-2.6 (reject/flag answers
whose citations don't hold up) from SRS.md. This does NOT trust the model's
self-reported citations — every citation is checked against:
  1. the actual set of chunk_ids that were retrieved (did it cite something real?)
  2. the chunk's actual text (did it quote something that's actually in there?)

Never trust the LLM's own claim of compliance — verify server-side (SRS section 9, Risks).
"""
import json
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

from src.retrieval.types import RetrievedChunk

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a support assistant. Answer the user's question using ONLY the
provided context chunks.

- If the context has information relevant to the question, use it — even if it doesn't
  give the exact number or format the question implies (e.g. if pricing is "custom,
  contact sales" rather than a fixed number, say that instead of claiming you don't know).
- If the question assumes something the context contradicts or never mentions (a false
  premise), say so explicitly using what the context actually says. This counts as
  successfully answering — set "answerable" to true, since you gave the user a real,
  useful, grounded answer (that the premise is false), not a non-answer.
- Set "answerable" to false ONLY if the context contains nothing relevant at all, such
  that you cannot say anything useful about the question.
- Every factual claim in your answer must have a matching entry in "citations": the
  chunk_id it came from, and a short VERBATIM quote copied exactly from that chunk's text
  that supports the claim. Do not paraphrase the quote — copy it exactly.
- Never fabricate facts, numbers, or fees not present in the context."""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounded_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answerable": {
                    "type": "boolean",
                    "description": (
                        "true if you were able to give a real, useful, grounded answer — "
                        "including correcting a false premise using the context. false only "
                        "if the context has nothing relevant to say about the question at all."
                    ),
                },
                "answer": {
                    "type": "string",
                    "description": "the answer, or an explanation of what's missing if not answerable",
                },
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {"type": "string"},
                            "supporting_quote": {
                                "type": "string",
                                "description": "verbatim quote copied exactly from the cited chunk's text",
                            },
                        },
                        "required": ["chunk_id", "supporting_quote"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["answerable", "answer", "citations"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class InvalidCitation:
    chunk_id: str
    reason: str
    supporting_quote: str = ""


@dataclass
class GroundedAnswer:
    answer: str
    answerable: bool
    citations: list[dict]
    citations_valid: bool
    invalid_citations: list[InvalidCitation] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)


def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
    return OpenAI(api_key=api_key)


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = [f"[chunk_id: {c.chunk_id}]\n{c.text}" for c in chunks]
    return "\n\n---\n\n".join(parts)


_WRAPPING_QUOTE_CHARS = "\"'“”‘’"
_MARKDOWN_LINE_PREFIX_RE = re.compile(r"(?m)^\s{0,3}(#{1,6}|[-*+])\s+")


def _normalize_for_comparison(text: str) -> str:
    """Strip markdown noise that changes formatting but not meaning, then
    collapse whitespace and wrapping quote characters, before comparing a
    citation's quote against the source chunk text.

    Source chunks are raw markdown/PDF/DOCX text — a model quoting the
    substance of "**Enterprise**: ..." as "Enterprise: ..." (dropping bold
    markers) or copying across a line-wrapped sentence hasn't fabricated
    anything, but a naive substring check would flag both as invalid. None
    of these transformations touch actual words — only markup and whitespace.
    """
    text = text.replace("**", "").replace("__", "")
    text = _MARKDOWN_LINE_PREFIX_RE.sub("", text)
    normalized = " ".join(text.split())
    return normalized.strip(_WRAPPING_QUOTE_CHARS)


def validate_citations(
    citations: list[dict], retrieved_chunks: list[RetrievedChunk]
) -> tuple[bool, list[InvalidCitation]]:
    chunk_text_by_id = {c.chunk_id: c.text for c in retrieved_chunks}
    invalid: list[InvalidCitation] = []

    for citation in citations:
        chunk_id = citation.get("chunk_id", "")
        quote = citation.get("supporting_quote", "")

        if chunk_id not in chunk_text_by_id:
            invalid.append(
                InvalidCitation(
                    chunk_id=chunk_id,
                    reason="chunk_id was not among the chunks actually retrieved for this query",
                    supporting_quote=quote,
                )
            )
            continue

        normalized_quote = _normalize_for_comparison(quote).lower()
        normalized_chunk = _normalize_for_comparison(chunk_text_by_id[chunk_id]).lower()
        if normalized_quote not in normalized_chunk:
            invalid.append(
                InvalidCitation(
                    chunk_id=chunk_id,
                    reason="supporting_quote does not appear verbatim in the cited chunk's text",
                    supporting_quote=quote,
                )
            )

    return len(invalid) == 0, invalid


def generate_grounded_answer(query: str, chunks: list[RetrievedChunk]) -> GroundedAnswer:
    client = get_openai_client()
    context_block = build_context_block(chunks)
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format=RESPONSE_SCHEMA,
    )

    parsed = json.loads(response.choices[0].message.content)
    citations_valid, invalid_citations = validate_citations(parsed["citations"], chunks)

    answer = parsed["answer"]
    if not citations_valid:
        # Hard gate (FR-2.6): don't silently serve an answer with bad citations.
        answer = (
            f"{answer}\n\n[WARNING: {len(invalid_citations)} citation(s) failed server-side "
            "validation — this answer may not be fully grounded in the retrieved context.]"
        )

    return GroundedAnswer(
        answer=answer,
        answerable=parsed["answerable"],
        citations=parsed["citations"],
        citations_valid=citations_valid,
        invalid_citations=invalid_citations,
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
    )
