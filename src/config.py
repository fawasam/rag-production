"""Retrieval/rerank tunables, overridable via environment variables (FR-2.7).

Defaults are reasonable for a small corpus — tune as the corpus grows.
"""
import os

DENSE_TOP_K = int(os.environ.get("DENSE_TOP_K", 10))
BM25_TOP_K = int(os.environ.get("BM25_TOP_K", 10))
RRF_K = int(os.environ.get("RRF_K", 60))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", 10))  # fused candidates fed to the cross-encoder
RERANK_TOP_M = int(os.environ.get("RERANK_TOP_M", 5))  # final chunks fed to the LLM
