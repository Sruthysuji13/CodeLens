"""FastAPI application providing REST endpoints for repository ingestion,
code retrieval, secret masking, grounded Q&A, and serving the frontend UI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ingestion import clone_or_open_repository, ingest_repository
from app.indexing import (
    metadata_store,
    vector_store,
    bm25_manager,
    index_repository_data,
)
from app.retrieval import HybridRetriever
from app.llm import answer_codebase_query, get_llm_client

# Application setup
app = FastAPI(
    title="Secure Software Engineering RAG Assistant",
    description="Grounded codebase question-answering with tree-sitter chunking, hybrid retrieval, and secret masking.",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = HybridRetriever(vector_store=vector_store, bm25_manager=bm25_manager)

# In-memory indexing status tracking
indexing_status = {
    "is_indexing": False,
    "current_repo": "",
    "message": "Idle",
    "error": None,
}


# Request & Response Models
class IndexRequest(BaseModel):
    repo_url_or_path: str = Field(..., description="GitHub repository URL (https://github.com/...) or local directory path")


class QueryRequest(BaseModel):
    repo_name: str = Field(..., description="Target repository name")
    query: str = Field(..., description="Natural language question about the codebase")
    top_k: int = Field(default=5, description="Number of source chunks to retrieve")


# Endpoints
@app.get("/api/status")
def get_system_status() -> dict[str, Any]:
    """Return backend status, LLM mode, and current indexing status."""
    grok_key_configured = bool(os.getenv("GROK_API_KEY", "").strip())
    return {
        "status": "online",
        "llm_provider": "Grok (xAI)" if grok_key_configured else "Mock / Offline Synthesizer",
        "embedding_model": os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
        "indexing_state": indexing_status,
    }


@app.get("/api/repositories")
def list_repositories() -> list[dict[str, Any]]:
    """List all indexed repositories with metadata."""
    return metadata_store.list_repositories()


@app.post("/api/repositories/index")
def index_repository(request: IndexRequest) -> dict[str, Any]:
    """Clone, parse, and index a GitHub repository synchronously or track progress."""
    global indexing_status
    if indexing_status["is_indexing"]:
        raise HTTPException(status_code=409, detail="Another indexing task is currently running.")

    repo_url = request.repo_url_or_path.strip()
    if not repo_url:
        raise HTTPException(status_code=400, detail="Repository URL or path is required.")

    indexing_status["is_indexing"] = True
    indexing_status["error"] = None
    indexing_status["message"] = f"Cloning repository: {repo_url}"

    try:
        # 1. Clone or open repo
        repo_name, local_path, commit_hash = clone_or_open_repository(repo_url)
        indexing_status["current_repo"] = repo_name
        indexing_status["message"] = f"Parsing source code for {repo_name}..."

        # 2. Ingest and extract tree-sitter chunks
        chunks, indexed_files = ingest_repository(local_path, repo_name)
        if not chunks:
            raise ValueError(f"No parseable source files found in repository {repo_name}.")

        # 3. Store and index in SQLite, ChromaDB, and BM25
        indexing_status["message"] = f"Vectorizing and indexing {len(chunks)} chunks..."
        repo_info = index_repository_data(
            repo_name=repo_name,
            repo_url=repo_url,
            commit_hash=commit_hash,
            chunks=chunks,
            file_paths=indexed_files,
        )

        indexing_status["message"] = f"Successfully indexed {repo_name}."
        return {
            "status": "success",
            "repository": repo_info,
            "total_files": len(indexed_files),
            "total_chunks": len(chunks),
        }
    except Exception as e:
        indexing_status["error"] = str(e)
        indexing_status["message"] = f"Failed to index: {str(e)}"
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        indexing_status["is_indexing"] = False


@app.get("/api/repositories/{repo_name}/files")
def get_repository_files(repo_name: str) -> list[str]:
    """List all indexed file paths for a repository."""
    files = metadata_store.get_files_for_repo(repo_name)
    if not files:
        raise HTTPException(status_code=404, detail=f"No files found for repository '{repo_name}'.")
    return files


@app.get("/api/repositories/{repo_name}/file")
def get_file_content(repo_name: str, path: str = Query(..., description="Relative file path")) -> dict[str, Any]:
    """Get the source code content of a specific repository file."""
    repo = metadata_store.get_repository(repo_name)
    if not repo:
        raise HTTPException(status_code=404, detail=f"Repository '{repo_name}' not found.")

    repo_url_or_path = repo.get("url", "")
    if os.path.isdir(repo_url_or_path):
        local_repo_dir = os.path.abspath(repo_url_or_path)
    else:
        local_repo_dir = os.path.abspath(os.path.join(os.getenv("REPOSITORIES_DIR", "data/repositories"), repo_name))

    target_file = os.path.abspath(os.path.join(local_repo_dir, path))

    # Path traversal protection
    if not target_file.startswith(local_repo_dir):
        raise HTTPException(status_code=400, detail="Invalid file path.")

    if not os.path.isfile(target_file):
        raise HTTPException(status_code=404, detail=f"File '{path}' does not exist.")

    try:
        with open(target_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return {
            "repo_name": repo_name,
            "file_path": path,
            "total_lines": len(lines),
            "content": "".join(lines),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")


@app.post("/api/query")
def query_repository(request: QueryRequest) -> dict[str, Any]:
    """Answer a natural-language question grounded in the repository context."""
    repo_name = request.repo_name.strip()
    query = request.query.strip()

    if not repo_name or not query:
        raise HTTPException(status_code=400, detail="Both 'repo_name' and 'query' are required.")

    # Check if repository exists
    repo = metadata_store.get_repository(repo_name)
    if not repo:
        raise HTTPException(status_code=404, detail=f"Repository '{repo_name}' is not indexed.")

    # 1. Retrieve hybrid chunks (semantic + BM25 + RRF + reranking)
    top_chunks = retriever.retrieve(query=query, repo_name=repo_name, top_k=request.top_k)

    # 2. Mask secrets and generate grounded answer
    result = answer_codebase_query(query=query, retrieved_chunks=top_chunks)
    result["repo_name"] = repo_name
    return result


# Static files and root UI route
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Secure Software Engineering RAG Assistant</h1><p>Frontend static files not found.</p>")

