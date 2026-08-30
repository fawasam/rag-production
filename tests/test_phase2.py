"""Phase 2 end-to-end eval harness — same golden_set.jsonl as Phase 1, but
exercised through the full hybrid pipeline: dense + BM25 -> RRF -> cross-encoder
rerank -> schema-enforced, server-validated citations.

Checks, per case:
  1. Retrieval hit@k (post-rerank)         — did the expected chunk(s) survive?
  2. Answer content                          — does the answer say the right thing?
  3. Abstention                              — for unanswerable questions, is the
                                                answer non-committal instead of a
                                                confident guess?
  4. Citation validity (hard gate, FR-2.6)   — did every citation pass server-side
                                                validation against the real chunk text?

Note on the "answerable" field: it's logged for every case but NOT hard-asserted
for answerable questions. In practice gpt-4o-mini sets it inconsistently for
negative-polarity answers ("No, X does not support Y") and false-premise
corrections — sometimes False even when the answer text is fully correct and
well-cited. That's a real, observed soft-signal reliability gap (exactly what
Phase 3 observability is meant to track over time), not something to paper over
by re-prompting until the flag happens to match on this corpus. The content
checks (must_contain / abstention phrasing) are the actual correctness signal.

Run:
    pytest tests/test_phase2.py -v -s
"""
import json
import os
from pathlib import Path

import pytest

from src.generation.grounded_client import generate_grounded_answer
from src.retrieval.hybrid import retrieve_hybrid

GOLDEN_SET_PATH = Path(__file__).resolve().parents[1] / "eval" / "golden_set.jsonl"

ABSTENTION_PHRASES = [
    "don't know",
    "do not know",
    "not sure",
    "no information",
    "not mentioned",
    "does not mention",
    "cannot find",
    "can't find",
    "not specified",
    "not provided",
    "doesn't contain",
    "does not contain",
    "no details",
    "unable to find",
]

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — copy .env.example to .env",
)


def load_golden_set():
    cases = []
    with open(GOLDEN_SET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


@pytest.mark.parametrize("case", load_golden_set(), ids=lambda c: c["id"])
def test_golden_case_hybrid(case):
    rerank_top_m = case.get("top_k", 5)  # reuse Phase 1's per-case override, if any
    debug = retrieve_hybrid(case["question"], rerank_top_n=10, rerank_top_m=rerank_top_m)
    retrieved_ids = [c.chunk_id for c in debug.reranked_results]

    result = generate_grounded_answer(case["question"], debug.reranked_results)
    answer_lower = result.answer.lower()

    print(f"\nQ: {case['question']}")
    if case.get("note"):
        print(f"Why this is hard: {case['note']}")
    print(f"Dense:     {[c.chunk_id for c in debug.dense_results]}")
    print(f"BM25:      {[c.chunk_id for c in debug.bm25_results]}")
    print(f"Reranked:  {retrieved_ids}")
    print(f"Answerable: {result.answerable} | Citations valid: {result.citations_valid}")
    print(f"A: {result.answer}")

    # Hard gate (FR-2.6): a well-behaved run should never fail citation validation.
    assert result.citations_valid, (
        f"Citation validation failed — model cited something not actually in the "
        f"retrieved context: {result.invalid_citations}"
    )

    if case["unanswerable"]:
        abstained = not result.answerable or any(
            phrase in answer_lower for phrase in ABSTENTION_PHRASES
        )
        assert abstained, (
            f"Expected the model to abstain (question isn't in the corpus), "
            f"but it answered confidently: {result.answer!r}"
        )
        return

    if case.get("expected_chunk_id"):
        assert case["expected_chunk_id"] in retrieved_ids, (
            f"Expected chunk {case['expected_chunk_id']!r} not in reranked "
            f"top-{rerank_top_m}: {retrieved_ids}"
        )

    for expected_id in case.get("expected_chunk_ids", []):
        assert expected_id in retrieved_ids, (
            f"Multi-hop question needs chunk {expected_id!r} but reranked "
            f"top-{rerank_top_m} was only: {retrieved_ids}"
        )

    for substring in case.get("must_contain", []):
        assert substring.lower() in answer_lower, (
            f"Expected answer to contain {substring!r} but got: {result.answer!r}"
        )

    any_of = case.get("must_contain_any", [])
    if any_of:
        assert any(s.lower() in answer_lower for s in any_of), (
            f"Expected answer to contain at least one of {any_of!r} but got: {result.answer!r}"
        )

    for substring in case.get("must_not_contain", []):
        assert substring.lower() not in answer_lower, (
            f"Answer should NOT contain {substring!r}. Got: {result.answer!r}"
        )
