import pytest
from app.ingestion import (
    extract_tree_sitter_chunks,
    extract_markdown_chunks,
    extract_structured_data_chunks,
)


def test_python_tree_sitter_chunking():
    code = """class UserService:
    def __init__(self, db):
        self.db = db

    def authenticate(self, username, password):
        if not username:
            return False
        return True


def helper_function(x):
    return x * 2
"""
    chunks = extract_tree_sitter_chunks(code, "python", "test-repo", "src/services/user_service.py")
    assert len(chunks) >= 3

    symbols = [c["symbol_name"] for c in chunks]
    assert "UserService" in symbols
    assert "UserService.authenticate" in symbols
    assert "helper_function" in symbols

    # Check line numbers and file path
    auth_chunk = next(c for c in chunks if c["symbol_name"] == "UserService.authenticate")
    assert auth_chunk["file_path"] == "src/services/user_service.py"
    assert auth_chunk["start_line"] == 5
    assert auth_chunk["end_line"] == 8


def test_java_tree_sitter_chunking():
    code = """package com.example.service;

public class AuthService {
    private final TokenProvider tokenProvider;

    public AuthService(TokenProvider tokenProvider) {
        this.tokenProvider = tokenProvider;
    }

    public String login(String username, String password) {
        return tokenProvider.generate(username);
    }
}
"""
    chunks = extract_tree_sitter_chunks(code, "java", "test-repo", "src/security/AuthService.java")
    assert len(chunks) >= 2

    symbols = [c["symbol_name"] for c in chunks]
    assert "AuthService" in symbols
    assert "AuthService.login" in symbols

    login_chunk = next(c for c in chunks if c["symbol_name"] == "AuthService.login")
    assert login_chunk["start_line"] == 10
    assert login_chunk["end_line"] == 12


def test_typescript_tree_sitter_chunking():
    code = """export interface UserProfile {
    id: string;
    username: string;
}

export function validateUser(user: UserProfile): boolean {
    return Boolean(user.id);
}
"""
    chunks = extract_tree_sitter_chunks(code, "typescript", "test-repo", "src/types/user.ts")
    assert len(chunks) >= 2
    symbols = [c["symbol_name"] for c in chunks]
    assert "UserProfile" in symbols
    assert "validateUser" in symbols


def test_c_cpp_tree_sitter_chunking():
    c_code = """#include <stdio.h>

int compute_hash(const char* input) {
    return 42;
}
"""
    chunks = extract_tree_sitter_chunks(c_code, "c", "test-repo", "src/crypto/hash.c")
    assert len(chunks) >= 1
    symbols = [c["symbol_name"] for c in chunks]
    assert any("compute_hash" in s for s in symbols)


def test_markdown_chunking():
    md = """# Architecture Overview
This is the system architecture.

## Authentication Flow
Users authenticate via OAuth2.

## Database Design
PostgreSQL is used.
"""
    chunks = extract_markdown_chunks(md, "test-repo", "docs/ARCH.md")
    assert len(chunks) == 3
    symbols = [c["symbol_name"] for c in chunks]
    assert "Architecture Overview" in symbols
    assert "Authentication Flow" in symbols
    assert "Database Design" in symbols
    assert chunks[1]["start_line"] == 4


def test_structured_data_chunking():
    json_data = '{\n  "service": "auth",\n  "port": 8080\n}'
    chunks = extract_structured_data_chunks(json_data, "json", "test-repo", "config.json")
    assert len(chunks) == 1
    assert chunks[0]["file_path"] == "config.json"
    assert chunks[0]["language"] == "json"

