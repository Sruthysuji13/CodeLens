"""Seed script that creates a ready-to-use sample repository in data/repositories/sample-ecommerce."""

import os
import sys
import git
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import ingest_repository
from app.indexing import index_repository_data


def seed_sample_repo():
    base_dir = Path(__file__).resolve().parent.parent
    repo_dir = base_dir / "data" / "repositories" / "sample-ecommerce"
    repo_dir_str = str(repo_dir)

    os.makedirs(repo_dir_str, exist_ok=True)
    try:
        repo = git.Repo(repo_dir_str)
    except Exception:
        repo = git.Repo.init(repo_dir_str)

    # 1. Java AuthService
    sec_dir = repo_dir / "src" / "main" / "java" / "com" / "example" / "security"
    sec_dir.mkdir(parents=True, exist_ok=True)
    auth_file = sec_dir / "AuthService.java"
    auth_file.write_text(
        """package com.example.security;

public class AuthService {
    // Database connection string with embedded password (used for secret masking test)
    private final String dbUri = "postgresql://db_user:s3cr3tP@ssw0rd!@postgres.internal:5432/auth_db";
    private final TokenValidator tokenValidator;

    public AuthService(TokenValidator tokenValidator) {
        this.tokenValidator = tokenValidator;
    }

    public boolean validateToken(String token) {
        if (token == null || token.isEmpty()) {
            return false;
        }
        return tokenValidator.isValid(token);
    }

    public String generateToken(String username) {
        return "jwt." + username + ".signature";
    }
}
""",
        encoding="utf-8",
    )

    # 2. Java LoginController
    ctrl_dir = repo_dir / "src" / "main" / "java" / "com" / "example" / "controller"
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    login_file = ctrl_dir / "LoginController.java"
    login_file.write_text(
        """package com.example.controller;

import com.example.security.AuthService;

public class LoginController {
    private final AuthService authService;

    public LoginController(AuthService authService) {
        this.authService = authService;
    }

    public Response handleLogin(Request request) {
        String token = request.getHeader("Authorization");
        if (token == null || !authService.validateToken(token)) {
            return Response.status(401).entity("Invalid or expired authorization token").build();
        }
        return Response.ok("Login successful").build();
    }
}
""",
        encoding="utf-8",
    )

    # 3. Python UserService
    svc_dir = repo_dir / "services"
    svc_dir.mkdir(parents=True, exist_ok=True)
    user_file = svc_dir / "user_service.py"
    user_file.write_text(
        '''"""User management and persistence service."""

class UserService:
    def __init__(self, db_client):
        self.db = db_client

    def find_user_by_id(self, user_id: str):
        return self.db.query("SELECT * FROM users WHERE id = ?", (user_id,))

    def authenticate_credentials(self, username: str, password_hash: str) -> bool:
        user = self.db.find_one({"username": username})
        if not user or user.get("password_hash") != password_hash:
            return False
        return True
''',
        encoding="utf-8",
    )

    # 4. TypeScript Auth Client
    ts_dir = repo_dir / "frontend" / "src"
    ts_dir.mkdir(parents=True, exist_ok=True)
    client_file = ts_dir / "authClient.ts"
    client_file.write_text(
        """export interface AuthResponse {
    token: string;
    expiresIn: number;
}

export async function loginWithCredentials(username: string, pass: string): Promise<AuthResponse> {
    const res = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, pass })
    });
    if (res.status === 401) {
        throw new Error("Unauthorized: Check your credentials or token");
    }
    return res.json();
}
""",
        encoding="utf-8",
    )

    # 5. Architecture Documentation
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    arch_file = docs_dir / "ARCHITECTURE.md"
    arch_file.write_text(
        """# E-Commerce Architecture

## Authentication System
Clients authenticate by sending credentials or Bearer tokens to `LoginController`.
`LoginController` delegates token validation to `AuthService.validateToken`.
If the token is missing or invalid, the controller immediately returns HTTP 401.

## Service Dependencies
- `LoginController` depends on `AuthService`.
- If `UserService` changes its schema or query interface, user profile endpoints and `LoginController` will be impacted.
""",
        encoding="utf-8",
    )

    # Commit repository
    try:
        repo.git.add(A=True)
        repo.index.commit("Initial demo repository commit")
    except Exception:
        pass

    chunks, indexed_files = ingest_repository(repo_dir_str, "sample-ecommerce")
    repo_info = index_repository_data(
        repo_name="sample-ecommerce",
        repo_url=repo_dir_str,
        commit_hash="demo_hash",
        chunks=chunks,
        file_paths=indexed_files,
    )
    print(f"Successfully seeded 'sample-ecommerce' ({len(indexed_files)} files, {len(chunks)} chunks).")
    return repo_info


if __name__ == "__main__":
    seed_sample_repo()
