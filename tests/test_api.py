import os
import git
import pytest
from fastapi.testclient import TestClient
from app.api import app
from app.indexing import metadata_store

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_git_repo(tmp_path_factory):
    """Create a temporary git repository with Python and Java files for testing."""
    repo_dir = tmp_path_factory.mktemp("test_sample_repo")
    repo = git.Repo.init(str(repo_dir))

    # Create Python file
    src_dir = repo_dir / "src"
    src_dir.mkdir()
    user_service_file = src_dir / "user_service.py"
    user_service_file.write_text(
        """class UserService:
    def __init__(self, db_client):
        self.db = db_client

    def authenticate(self, username, password):
        if not username or not password:
            return False
        return True
""",
        encoding="utf-8",
    )

    # Create Java file with an obvious secret
    auth_file = src_dir / "AuthService.java"
    dummy_key = "sk-" + "proj1234567890abcdef1234567890"
    auth_file.write_text(
        f"""package com.example;

public class AuthService {{
    private String apiKey = "{dummy_key}";

    public boolean login(String user, String pass) {{
        return user != null && pass != null;
    }}
}}
""",
        encoding="utf-8",
    )

    # Commit files
    repo.index.add([str(user_service_file), str(auth_file)])
    repo.index.commit("Initial commit")

    return str(repo_dir)


def test_get_status():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "llm_provider" in data


def test_serve_ui():
    res = client.get("/")
    assert res.status_code == 200
    assert "Secure Software Engineering RAG Assistant" in res.text


def test_index_and_query_repository(sample_git_repo):
    # 1. Index the sample repository
    index_res = client.post("/api/repositories/index", json={"repo_url_or_path": sample_git_repo})
    assert index_res.status_code == 200
    repo_data = index_res.json()
    assert repo_data["status"] == "success"
    repo_name = repo_data["repository"]["name"]

    # 2. List repositories
    list_res = client.get("/api/repositories")
    assert list_res.status_code == 200
    repos = list_res.json()
    assert any(r["name"] == repo_name for r in repos)

    # 3. Get repository files
    files_res = client.get(f"/api/repositories/{repo_name}/files")
    assert files_res.status_code == 200
    files = files_res.json()
    assert any("user_service.py" in f for f in files)
    assert any("AuthService.java" in f for f in files)

    # 4. Read file content
    content_res = client.get(f"/api/repositories/{repo_name}/file?path=src/user_service.py")
    assert content_res.status_code == 200
    assert "class UserService:" in content_res.json()["content"]

    # 5. Query the repository for authentication
    query_res = client.post(
        "/api/query",
        json={"repo_name": repo_name, "query": "How does authentication work?", "top_k": 3},
    )
    assert query_res.status_code == 200
    result = query_res.json()
    assert "answer" in result
    assert len(result["citations"]) >= 1
    # Check that secrets in AuthService were masked
    assert result["secrets_masked_count"] >= 1

