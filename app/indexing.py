"""Indexing and storage module for SQLite metadata, ChromaDB vector database, and BM25 search.

Stores repository details, code chunks, symbol metadata, and line numbers.
Provides vector embedding indexing via BAAI/bge-small-en-v1.5 and keyword indexing via rank-bm25.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Ensure offline operation for downloaded models
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import chromadb
from rank_bm25 import BM25Okapi, BM25Plus
from sentence_transformers import SentenceTransformer

# Default paths
DB_PATH = os.getenv("DATABASE_PATH", "data/database/metadata.db")
CHROMA_PATH = os.getenv("CHROMA_DIR", "data/chroma")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# Singleton embedding model holder
_EMBEDDING_MODEL: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or lazily initialize the SentenceTransformer embedding model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


# SQLite Storage Manager
class SQLiteMetadataStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Create tables if they don't already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS repositories (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE,
                    url TEXT,
                    commit_hash TEXT,
                    indexed_at TEXT,
                    total_files INTEGER,
                    total_chunks INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    repo_name TEXT,
                    file_path TEXT,
                    language TEXT,
                    chunk_type TEXT,
                    symbol_name TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    content TEXT,
                    FOREIGN KEY (repo_name) REFERENCES repositories (name) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks (repo_name)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks (file_path)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks (symbol_name)
            """)
            conn.commit()

    def save_repository(
        self,
        repo_name: str,
        repo_url: str,
        commit_hash: str,
        total_files: int,
        total_chunks: int,
    ) -> dict[str, Any]:
        """Insert or replace repository entry."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO repositories (id, name, url, commit_hash, indexed_at, total_files, total_chunks)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    url=excluded.url,
                    commit_hash=excluded.commit_hash,
                    indexed_at=excluded.indexed_at,
                    total_files=excluded.total_files,
                    total_chunks=excluded.total_chunks
                """,
                (repo_name, repo_name, repo_url, commit_hash, now, total_files, total_chunks),
            )
            conn.commit()

        return {
            "name": repo_name,
            "url": repo_url,
            "commit_hash": commit_hash,
            "indexed_at": now,
            "total_files": total_files,
            "total_chunks": total_chunks,
        }

    def save_chunks(self, repo_name: str, chunks: list[dict[str, Any]]):
        """Batch save chunks into SQLite, removing any previous chunks for this repo."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE repo_name = ?", (repo_name,))

            insert_rows = [
                (
                    c["id"],
                    repo_name,
                    c["file_path"],
                    c["language"],
                    c["chunk_type"],
                    c.get("symbol_name", ""),
                    c["start_line"],
                    c["end_line"],
                    c["content"],
                )
                for c in chunks
            ]
            cursor.executemany(
                """
                INSERT OR REPLACE INTO chunks 
                (id, repo_name, file_path, language, chunk_type, symbol_name, start_line, end_line, content)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )
            conn.commit()

    def get_repository(self, repo_name: str) -> Optional[dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM repositories WHERE name = ?", (repo_name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_repositories(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM repositories ORDER BY indexed_at DESC")
            return [dict(r) for r in cursor.fetchall()]

    def get_chunks_for_repo(self, repo_name: str) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE repo_name = ? ORDER BY file_path, start_line", (repo_name,))
            return [dict(r) for r in cursor.fetchall()]

    def get_files_for_repo(self, repo_name: str) -> list[str]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT file_path FROM chunks WHERE repo_name = ? ORDER BY file_path", (repo_name,))
            return [r["file_path"] for r in cursor.fetchall()]


# Chroma Vector DB Manager
class ChromaVectorStore:
    def __init__(self, persist_directory: str = CHROMA_PATH):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="code_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, repo_name: str, chunks: list[dict[str, Any]], batch_size: int = 128):
        """Compute embeddings and store chunks into ChromaDB."""
        if not chunks:
            return

        # Delete existing entries for this repo from Chroma
        try:
            self.collection.delete(where={"repo_name": repo_name})
        except Exception:
            pass

        model = get_embedding_model()

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["content"] for c in batch]
            ids = [c["id"] for c in batch]
            metadatas = [
                {
                    "repo_name": repo_name,
                    "file_path": c["file_path"],
                    "language": c["language"],
                    "chunk_type": c["chunk_type"],
                    "symbol_name": c.get("symbol_name", "") or "",
                    "start_line": int(c["start_line"]),
                    "end_line": int(c["end_line"]),
                }
                for c in batch
            ]

            embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    def query_semantic(self, query: str, repo_name: str, top_k: int = 15) -> list[dict[str, Any]]:
        """Dense semantic search using vector similarity."""
        model = get_embedding_model()
        query_embedding = model.encode([query], show_progress_bar=False, normalize_embeddings=True).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={"repo_name": repo_name},
            include=["documents", "metadatas", "distances"],
        )

        semantic_hits = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for idx, chunk_id in enumerate(ids):
            meta = metas[idx] if idx < len(metas) else {}
            doc = docs[idx] if idx < len(docs) else ""
            dist = distances[idx] if idx < len(distances) else 1.0
            # Cosine distance to similarity: 1 - distance
            similarity = max(0.0, 1.0 - dist)

            semantic_hits.append({
                "id": chunk_id,
                "content": doc,
                "repo_name": meta.get("repo_name", repo_name),
                "file_path": meta.get("file_path", ""),
                "language": meta.get("language", ""),
                "chunk_type": meta.get("chunk_type", ""),
                "symbol_name": meta.get("symbol_name", ""),
                "start_line": meta.get("start_line", 1),
                "end_line": meta.get("end_line", 1),
                "semantic_score": similarity,
            })

        return semantic_hits


# Code-aware Tokenizer & BM25 Manager
def code_tokenize(text: str) -> list[str]:
    """Tokenize code text, splitting snake_case, camelCase, and punctuation."""
    # Split camelCase: UserService -> User Service
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    # Split non-alphanumeric characters
    tokens = re.findall(r"[A-Za-z0-9]+", s1)
    # Lowercase and retain meaningful tokens
    result = []
    for tok in tokens:
        low = tok.lower()
        if len(low) > 1:
            result.append(low)
    return result


class BM25IndexManager:
    """Manages in-memory BM25 indexes per repository."""
    def __init__(self, metadata_store: SQLiteMetadataStore):
        self.metadata_store = metadata_store
        self._indexes: dict[str, Any] = {}
        self._chunk_lookup: dict[str, list[dict[str, Any]]] = {}
        self._corpus_tokens: dict[str, list[list[str]]] = {}

    def build_or_get_index(self, repo_name: str) -> tuple[Optional[Any], list[dict[str, Any]], list[list[str]]]:
        if repo_name in self._indexes:
            return self._indexes[repo_name], self._chunk_lookup[repo_name], self._corpus_tokens[repo_name]

        chunks = self.metadata_store.get_chunks_for_repo(repo_name)
        if not chunks:
            return None, [], []

        corpus_tokens = [code_tokenize(c["content"] + " " + c["file_path"] + " " + c.get("symbol_name", "")) for c in chunks]
        if len(corpus_tokens) <= 3:
            bm25 = BM25Plus(corpus_tokens)
        else:
            bm25 = BM25Okapi(corpus_tokens)

        self._indexes[repo_name] = bm25
        self._chunk_lookup[repo_name] = chunks
        self._corpus_tokens[repo_name] = corpus_tokens
        return bm25, chunks, corpus_tokens

    def invalidate(self, repo_name: str):
        self._indexes.pop(repo_name, None)
        self._chunk_lookup.pop(repo_name, None)
        self._corpus_tokens.pop(repo_name, None)

    def search_keyword(self, query: str, repo_name: str, top_k: int = 15) -> list[dict[str, Any]]:
        bm25, chunks, corpus_tokens = self.build_or_get_index(repo_name)
        if not bm25 or not chunks:
            return []

        tokenized_query = list(dict.fromkeys(code_tokenize(query)))
        if not tokenized_query:
            return []

        scores = bm25.get_scores(tokenized_query)
        scored_chunks = []
        q_set = set(tokenized_query)
        for idx, score in enumerate(scores):
            doc_terms = set(corpus_tokens[idx])
            matched = q_set.intersection(doc_terms)
            if matched and score > 0:
                chunk = dict(chunks[idx])
                chunk["bm25_score"] = float(score)
                chunk["matched_terms"] = list(matched)
                scored_chunks.append(chunk)

        scored_chunks.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_chunks[:top_k]


# Global storage singletons
metadata_store = SQLiteMetadataStore()
vector_store = ChromaVectorStore()
bm25_manager = BM25IndexManager(metadata_store)


def index_repository_data(
    repo_name: str,
    repo_url: str,
    commit_hash: str,
    chunks: list[dict[str, Any]],
    file_paths: list[str],
) -> dict[str, Any]:
    """Complete indexing pipeline: SQLite + ChromaDB + BM25 invalidation."""
    # 1. Save metadata in SQLite
    repo_info = metadata_store.save_repository(
        repo_name=repo_name,
        repo_url=repo_url,
        commit_hash=commit_hash,
        total_files=len(file_paths),
        total_chunks=len(chunks),
    )
    metadata_store.save_chunks(repo_name, chunks)

    # 2. Index in Chroma vector store
    vector_store.index_chunks(repo_name, chunks)

    # 3. Invalidate BM25 cache so next query builds fresh
    bm25_manager.invalidate(repo_name)

    return repo_info

