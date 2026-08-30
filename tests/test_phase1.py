"""Phase 1 end-to-end eval harness.

Runs every case in eval/golden_set.jsonl through retrieve() + generate_answer()
and checks:
  1. Retrieval hit@k  — did the expected chunk show up in top-k?
  2. Answer content   — does the answer contain/avoid the expected substrings?
  3. Abstention        — for questions the corpus can't answer, does the model
                         say so instead of making something up?

This is a manual precursor to the automated Ragas/TruLens gate in Phase 2
(see SRS.md section 8) — same golden_set.jsonl will be reused there.

Run:
    pytest tests/test_phase1.py -v -s
"""
import json
import os
from pathlib import Path

import pytest

from src.generation.llm_client import generate_answer
from src.retrieval.dense import retrieve

GOLDEN_SET_PATH = Path(__file__).resolve().parents[1] / "eval" / "golden_set.jsonl"
TOP_K = 3

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
def test_golden_case(case):
    chunks = retrieve(case["question"], top_k=TOP_K)
    retrieved_ids = [c.chunk_id for c in chunks]
    answer = generate_answer(case["question"], chunks)
    answer_lower = answer.lower()

    print(f"\nQ: {case['question']}")
    print(f"Retrieved: {retrieved_ids}")
    print(f"A: {answer}")

    if case["unanswerable"]:
        assert any(phrase in answer_lower for phrase in ABSTENTION_PHRASES), (
            f"Expected the model to abstain (question isn't in the corpus), "
            f"but it answered confidently: {answer!r}"
        )
        return

    if case.get("expected_chunk_id"):
        assert case["expected_chunk_id"] in retrieved_ids, (
            f"Expected chunk {case['expected_chunk_id']!r} not in top-{TOP_K} "
            f"retrieved: {retrieved_ids}"
        )

    for substring in case.get("must_contain", []):
        assert substring.lower() in answer_lower, (
            f"Expected answer to contain {substring!r} but got: {answer!r}"
        )

    for substring in case.get("must_not_contain", []):
        assert substring.lower() not in answer_lower, (
            f"Answer should NOT contain {substring!r} but got: {answer!r}"
        )
