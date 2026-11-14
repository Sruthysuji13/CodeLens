"""Hybrid retrieval module combining semantic search, BM25 keyword search,
Reciprocal Rank Fusion (RRF), and lightweight symbol/path reranking.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from app.indexing import (
    ChromaVectorStore,
    BM25IndexManager,
    code_tokenize,
    vector_store as default_vector_store,
    bm25_manager as default_bm25_manager,
)


def compute_rrf_scores(
    semantic_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    k: int = 60,
    semantic_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Combine dense semantic and sparse BM25 search results using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(d) = (w_sem / (k + rank_sem(d))) + (w_bm25 / (k + rank_bm25(d)))
    """
    merged_chunks: dict[str, dict[str, Any]] = {}
    rrf_scores: dict[str, float] = {}

    # Rank 1-based
    for rank, hit in enumerate(semantic_hits, start=1):
        cid = hit["id"]
        merged_chunks[cid] = dict(hit)
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (semantic_weight / (k + rank))
        merged_chunks[cid]["semantic_rank"] = rank

    for rank, hit in enumerate(bm25_hits, start=1):
        cid = hit["id"]
        if cid not in merged_chunks:
            merged_chunks[cid] = dict(hit)
        else:
            merged_chunks[cid]["bm25_score"] = hit.get("bm25_score", 0.0)
            merged_chunks[cid]["matched_terms"] = hit.get("matched_terms", [])
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (bm25_weight / (k + rank))
        merged_chunks[cid]["bm25_rank"] = rank

    results = []
    for cid, chunk in merged_chunks.items():
        chunk["rrf_score"] = rrf_scores[cid]
        results.append(chunk)

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


def lightweight_rerank(
    chunks: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """Apply lightweight code-aware reranking to fused chunks.

    Boosts chunks that:
    1. Have an exact or partial match with the queried symbol name.
    2. Have file paths matching entities in the query.
    3. Have high query token density in content.
    """
    query_tokens = set(code_tokenize(query))
    if not query_tokens:
        return chunks

    query_lower = query.lower()
    reranked = []

    for chunk in chunks:
        chunk_copy = dict(chunk)
        boost = 1.0

        symbol = (chunk_copy.get("symbol_name") or "").lower()
        file_path = (chunk_copy.get("file_path") or "").lower()
        content = chunk_copy.get("content", "").lower()

        # 1. Exact or partial symbol match
        if symbol:
            symbol_tokens = set(code_tokenize(symbol))
            overlap = query_tokens.intersection(symbol_tokens)
            if overlap:
                # Up to 35% boost for symbol overlap
                boost += 0.35 * (len(overlap) / max(1, len(symbol_tokens)))
            if symbol in query_lower or any(tok in symbol for tok in query_tokens if len(tok) >= 4):
                boost += 0.25

        # 2. File path match (e.g. UserService.java or user_service.py mentioned in query)
        file_name = file_path.split("/")[-1]
        file_base = file_name.split(".")[0]
        if file_base and (file_base in query_lower or file_base in query_tokens):
            boost += 0.30

        # 3. Query term coverage in chunk content
        content_tokens = set(code_tokenize(content[:1000]))
        term_coverage = len(query_tokens.intersection(content_tokens)) / len(query_tokens)
        boost += 0.20 * term_coverage

        chunk_copy["rerank_boost"] = round(boost, 4)
        chunk_copy["final_score"] = chunk_copy["rrf_score"] * boost
        reranked.append(chunk_copy)

    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked


class HybridRetriever:
    """Orchestrates hybrid semantic + BM25 retrieval, RRF fusion, and reranking."""

    def __init__(
        self,
        vector_store: ChromaVectorStore = default_vector_store,
        bm25_manager: BM25IndexManager = default_bm25_manager,
    ):
        self.vector_store = vector_store
        self.bm25_manager = bm25_manager

    def retrieve(
        self,
        query: str,
        repo_name: str,
        top_k: int = 5,
        candidate_pool_size: int = 15,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant code chunks for a user query.

        Args:
            query: Natural language or symbol query.
            repo_name: Indexed repository identifier.
            top_k: Final number of ranked chunks to return.
            candidate_pool_size: Number of candidates to fetch from each retriever.

        Returns:
            List of top-k ranked chunks with citation metadata.
        """
        # 1. Semantic retrieval (dense)
        try:
            semantic_hits = self.vector_store.query_semantic(query, repo_name, top_k=candidate_pool_size)
        except Exception:
            semantic_hits = []

        # 2. BM25 retrieval (sparse)
        try:
            bm25_hits = self.bm25_manager.search_keyword(query, repo_name, top_k=candidate_pool_size)
        except Exception:
            bm25_hits = []

        if not semantic_hits and not bm25_hits:
            return []

        # 3. Reciprocal Rank Fusion
        fused = compute_rrf_scores(semantic_hits, bm25_hits, k=60)

        # 4. Lightweight reranking
        reranked = lightweight_rerank(fused, query)

        # 5. Attach 1-based citation markers: [1], [2], ...
        final_top = reranked[:top_k]
        for idx, chunk in enumerate(final_top, start=1):
            chunk["citation_id"] = idx
            chunk["citation_label"] = (
                f"[{idx}] {chunk['file_path']} lines {chunk['start_line']}-{chunk['end_line']}"
            )

        return final_top


def format_context_for_prompt(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved and sanitized chunks into a structured context string for LLM."""
    if not chunks:
        return "No relevant source code snippets found."

    context_parts = []
    for c in chunks:
        citation = c.get("citation_label") or f"[{c.get('citation_id', 1)}] {c.get('file_path', 'unknown')}"
        symbol = f" (Symbol: {c['symbol_name']})" if c.get("symbol_name") else ""
        chunk_header = f"--- Source Citation: {citation}{symbol} ---"
        context_parts.append(f"{chunk_header}\n{c['content']}\n")

    return "\n".join(context_parts)

