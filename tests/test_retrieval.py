import pytest
from app.retrieval import (
    compute_rrf_scores,
    lightweight_rerank,
    format_context_for_prompt,
    HybridRetriever,
)


def test_compute_rrf_scores():
    semantic_hits = [
        {"id": "chunk_A", "symbol_name": "AuthService", "file_path": "Auth.java"},
        {"id": "chunk_B", "symbol_name": "UserService", "file_path": "User.java"},
    ]
    bm25_hits = [
        {"id": "chunk_B", "symbol_name": "UserService", "file_path": "User.java"},
        {"id": "chunk_C", "symbol_name": "OrderService", "file_path": "Order.java"},
    ]

    fused = compute_rrf_scores(semantic_hits, bm25_hits, k=60)
    # chunk_B appeared in both semantic (rank 2) and BM25 (rank 1), so its score:
    # 1/(60+2) + 1/(60+1) = 1/62 + 1/61 =~ 0.0161 + 0.01639 = 0.0325
    # chunk_A appeared only in semantic (rank 1): 1/(60+1) = 1/61 =~ 0.01639
    # chunk_C appeared only in BM25 (rank 2): 1/(60+2) = 1/62 =~ 0.0161
    # Thus chunk_B should be ranked #1!
    assert fused[0]["id"] == "chunk_B"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_lightweight_rerank():
    chunks = [
        {
            "id": "c1",
            "symbol_name": "TokenProvider",
            "file_path": "security/TokenProvider.java",
            "content": "class TokenProvider { String sign() {} }",
            "rrf_score": 0.02,
        },
        {
            "id": "c2",
            "symbol_name": "UserService.authenticate",
            "file_path": "service/UserService.java",
            "content": "class UserService { boolean authenticate(String u, String p) {} }",
            "rrf_score": 0.02,
        },
    ]

    reranked = lightweight_rerank(chunks, "How does UserService authenticate users?")
    # c2 has UserService and authenticate in both symbol, file_path, and content
    assert reranked[0]["id"] == "c2"
    assert reranked[0]["final_score"] > reranked[1]["final_score"]


def test_format_context_for_prompt():
    chunks = [
        {
            "citation_label": "[1] src/security/AuthService.java lines 42-68",
            "symbol_name": "AuthService.login",
            "content": "public String login(String u, String p) { return 'token'; }",
        }
    ]
    formatted = format_context_for_prompt(chunks)
    assert "[1] src/security/AuthService.java lines 42-68" in formatted
    assert "AuthService.login" in formatted
    assert "public String login" in formatted

