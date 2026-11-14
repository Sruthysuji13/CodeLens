import os
import pytest
from app.indexing import (
    SQLiteMetadataStore,
    ChromaVectorStore,
    BM25IndexManager,
    code_tokenize,
)


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_meta.db")
    return SQLiteMetadataStore(db_file)


@pytest.fixture
def temp_chroma(tmp_path):
    chroma_dir = str(tmp_path / "chroma_test")
    return ChromaVectorStore(chroma_dir)


def test_code_tokenize():
    tokens = code_tokenize("UserService.authenticateUser(loginRequest)")
    assert "user" in tokens
    assert "service" in tokens
    assert "authenticate" in tokens
    assert "login" in tokens
    assert "request" in tokens


def test_sqlite_store(temp_db):
    temp_db.save_repository("sample-repo", "https://github.com/org/sample-repo", "abc1234", 5, 10)
    repo = temp_db.get_repository("sample-repo")
    assert repo is not None
    assert repo["name"] == "sample-repo"
    assert repo["commit_hash"] == "abc1234"

    chunks = [
        {
            "id": "sample-repo:src/User.java:1:10",
            "file_path": "src/User.java",
            "language": "java",
            "chunk_type": "class",
            "symbol_name": "User",
            "start_line": 1,
            "end_line": 10,
            "content": "public class User { private String id; }",
        }
    ]
    temp_db.save_chunks("sample-repo", chunks)
    stored = temp_db.get_chunks_for_repo("sample-repo")
    assert len(stored) == 1
    assert stored[0]["symbol_name"] == "User"

    files = temp_db.get_files_for_repo("sample-repo")
    assert files == ["src/User.java"]


def test_bm25_manager(temp_db):
    chunks = [
        {
            "id": "repo:auth.py:1:10",
            "file_path": "auth.py",
            "language": "python",
            "chunk_type": "function",
            "symbol_name": "verify_jwt_token",
            "start_line": 1,
            "end_line": 10,
            "content": "def verify_jwt_token(token):\n    # verifies token\n    return True",
        },
        {
            "id": "repo:db.py:1:10",
            "file_path": "db.py",
            "language": "python",
            "chunk_type": "class",
            "symbol_name": "DatabasePool",
            "start_line": 1,
            "end_line": 10,
            "content": "class DatabasePool:\n    def connect(): pass",
        },
    ]
    temp_db.save_chunks("test-repo", chunks)
    manager = BM25IndexManager(temp_db)
    results = manager.search_keyword("verify jwt token", "test-repo", top_k=5)
    assert len(results) >= 1
    assert results[0]["symbol_name"] == "verify_jwt_token"


def test_chroma_vector_store(temp_chroma):
    chunks = [
        {
            "id": "test:auth.py:1:10",
            "file_path": "auth.py",
            "language": "python",
            "chunk_type": "function",
            "symbol_name": "authenticate_user",
            "start_line": 1,
            "end_line": 10,
            "content": "def authenticate_user(username, password):\n    # Checks credentials\n    return True",
        }
    ]
    temp_chroma.index_chunks("test-repo", chunks)
    results = temp_chroma.query_semantic("how do users log in and authenticate credentials", "test-repo", top_k=1)
    assert len(results) == 1
    assert results[0]["symbol_name"] == "authenticate_user"
    assert results[0]["semantic_score"] > 0.4

