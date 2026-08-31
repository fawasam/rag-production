"""LLM-judge implementations of the four RAG quality metrics from SRS.md
section 8: faithfulness, context precision, context recall, answer relevance.

Why hand-rolled instead of Ragas/TruLens: Ragas 0.4.3 (latest as of this
writing) is currently broken in two ways in this environment —
  1. A hard import of `langchain_community.chat_models.vertexai`, a module
     that no longer exists in current langchain-community (it was removed
     upstream; langchain-community is being sunset in favor of standalone
     integration packages). Worked around with a small compatibility shim,
     see the top of run_eval.py.
  2. Its `ascore()` calls hang indefinitely (confirmed >100s, no exception)
     even after the shim, isolated to ragas's instructor-wrapped LLM client
     specifically — a bare `AsyncOpenAI` call in the same environment
     completes in under a second. This looks like a real bug in ragas's
     async retry/instructor integration, not an environment or API-key issue.

Rather than keep debugging a third-party library's internals, these metrics
are implemented directly against the OpenAI client this project already uses
reliably everywhere else (same structured-output pattern as
generation/grounded_client.py). They follow Ragas's own metric *definitions*
(https://docs.ragas.io/en/stable/concepts/metrics/) closely enough to serve
the same CI-gating purpose. If ragas/TruLens gets fixed upstream, swapping
it back in means replacing this module's four functions — run_eval.py's
interface to them (question, answer, contexts, reference) -> 0..1 score
would not need to change.
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_MODEL = "gpt-4o-mini"

_SCORE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "metric_score",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "brief explanation for the score",
                },
                "score": {
                    "type": "number",
                    "description": "a score between 0.0 and 1.0",
                },
            },
            "required": ["reasoning", "score"],
            "additionalProperties": False,
        },
    },
}


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")
    return OpenAI(api_key=api_key)


def _judge(system_prompt: str, user_prompt: str) -> dict:
    client = _get_client()
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format=_SCORE_SCHEMA,
    )
    parsed = json.loads(response.choices[0].message.content)
    # Clamp defensively — the model is instructed to stay in [0, 1] but a
    # judge score feeding a CI gate should never trust that blindly.
    parsed["score"] = max(0.0, min(1.0, float(parsed["score"])))
    return parsed


def faithfulness(answer: str, contexts: list[str]) -> dict:
    """What fraction of the answer's claims are supported by the retrieved
    context? (Ragas definition: faithfulness = supported claims / total claims.)
    """
    context_block = "\n\n---\n\n".join(contexts)
    system_prompt = (
        "You are evaluating RAG faithfulness. Break the ANSWER down into its "
        "individual factual claims. For each claim, decide whether it is "
        "directly supported by the CONTEXT. Score = (number of supported "
        "claims) / (total number of claims), from 0.0 to 1.0. An answer that "
        "correctly says information is absent from the context, or correctly "
        "corrects a false premise using the context, counts as fully "
        "faithful (score 1.0) — it isn't asserting anything unsupported."
    )
    user_prompt = f"CONTEXT:\n{context_block}\n\nANSWER:\n{answer}"
    return _judge(system_prompt, user_prompt)


def context_precision(question: str, contexts: list[str], reference: str) -> dict:
    """Of the retrieved contexts, what fraction are actually relevant/useful
    for producing the reference answer? Penalizes noisy/irrelevant retrieval.
    """
    context_block = "\n\n---\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts))
    system_prompt = (
        "You are evaluating RAG context precision. Given the QUESTION and the "
        "REFERENCE (correct) answer, judge what fraction of the numbered "
        "CONTEXT chunks are actually relevant and useful for producing that "
        "reference answer. Score = (relevant chunks) / (total chunks), from "
        "0.0 to 1.0."
    )
    user_prompt = f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nCONTEXT:\n{context_block}"
    return _judge(system_prompt, user_prompt)


def context_recall(contexts: list[str], reference: str) -> dict:
    """Of the statements in the reference answer, what fraction can be
    attributed to (found within) the retrieved context? Penalizes missing
    information that should have been retrieved but wasn't.
    """
    context_block = "\n\n---\n\n".join(contexts)
    system_prompt = (
        "You are evaluating RAG context recall. Break the REFERENCE answer "
        "down into its individual statements. For each statement, decide "
        "whether it can be attributed to (is supported by) the CONTEXT. "
        "Score = (attributable statements) / (total statements), from 0.0 "
        "to 1.0."
    )
    user_prompt = f"REFERENCE ANSWER:\n{reference}\n\nCONTEXT:\n{context_block}"
    return _judge(system_prompt, user_prompt)


def answer_relevance(question: str, answer: str) -> dict:
    """How relevant/on-topic is the answer to the actual question asked —
    independent of whether it's factually correct (that's faithfulness's job).
    A correct-but-padded or partially-off-topic answer scores lower here.
    """
    system_prompt = (
        "You are evaluating RAG answer relevance. Judge how directly and "
        "completely the ANSWER addresses the QUESTION asked — not whether "
        "it's factually correct (a different metric), just whether it's "
        "on-topic, direct, and doesn't dodge or pad. Score 0.0 to 1.0."
    )
    user_prompt = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    return _judge(system_prompt, user_prompt)
