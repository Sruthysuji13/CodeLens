import pytest
from app.llm import (
    MockLLMClient,
    get_llm_client,
    answer_codebase_query,
)


def test_mock_llm_insufficient_evidence():
    client = MockLLMClient()
    answer = client.generate_answer("What is the meaning of life? Context: No relevant source code snippets found.")
    assert "Insufficient evidence from the repository." in answer


def test_answer_codebase_query_with_masking():
    retrieved_chunks = [
        {
            "citation_id": 1,
            "citation_label": "[1] src/security/AuthService.java lines 42-68",
            "file_path": "src/security/AuthService.java",
            "symbol_name": "AuthService.login",
            "start_line": 42,
            "end_line": 68,
            "language": "java",
            "content": 'String master_secret = "sk-superSecret1234567890abcdef"; // authenticate',
        }
    ]

    mock_client = MockLLMClient()
    result = answer_codebase_query("Explain authentication", retrieved_chunks, llm_client=mock_client)

    assert result["secrets_masked_count"] >= 1
    assert "sk-superSecret" not in result["citations"][0]["content"]
    assert (
        "[REDACTED_OPENAI_KEY]" in result["citations"][0]["content"]
        or "[REDACTED_SECRET]" in result["citations"][0]["content"]
    )
    assert len(result["citations"]) == 1
    assert "[1] src/security/AuthService.java lines 42-68" in result["answer"]

