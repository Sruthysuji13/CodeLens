"""Security module for detecting and masking secrets in source code and context snippets.

Practical masking layer designed to detect and redact API keys, tokens, passwords,
private keys, database credentials, connection strings, and obvious secrets before
they are sent to external LLMs.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class SecretDetection(NamedTuple):
    secret_type: str
    original_matched: str
    redacted_as: str
    start: int
    end: int


# Compiled regex patterns for various categories of secrets
SECRET_PATTERNS = [
    # 1. Private Keys
    (
        "PRIVATE_KEY",
        re.compile(
            r"-----BEGIN (?:[A-Z0-9_-]+ )?PRIVATE KEY(?: BLOCK)?-----[\s\S]*?-----END (?:[A-Z0-9_-]+ )?PRIVATE KEY(?: BLOCK)?-----",
            re.MULTILINE,
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    # 2. Database Connection Strings (e.g., postgresql://user:pass@host:port/db)
    (
        "DB_CONNECTION_STRING",
        re.compile(
            r"(?i)\b((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqps?|mssql):\/\/[^\s:@\"']+):([^\s@\"']+)@([^\s\"']+)\b"
        ),
        r"\1:[REDACTED_PASSWORD]@\3",
    ),
    # 3. AWS Credentials
    (
        "AWS_KEY_ID",
        re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
        "[REDACTED_AWS_KEY_ID]",
    ),
    (
        "AWS_SECRET",
        re.compile(
            r"(?i)(aws_secret_access_key|aws_session_token)\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?"
        ),
        r"\1=[REDACTED_AWS_SECRET]",
    ),
    # 4. Specific Provider API Keys
    (
        "OPENAI_API_KEY",
        re.compile(r"\b(sk-[a-zA-Z0-9_\-]{20,})\b"),
        "[REDACTED_OPENAI_KEY]",
    ),
    (
        "GITHUB_TOKEN",
        re.compile(r"\b(gh[pousr][_-][A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{22,})\b"),
        "[REDACTED_GITHUB_TOKEN]",
    ),
    (
        "SLACK_TOKEN",
        re.compile(r"\b(xox[baprs]-[0-9a-zA-Z-]{10,})\b"),
        "[REDACTED_SLACK_TOKEN]",
    ),
    (
        "GOOGLE_API_KEY",
        re.compile(r"\b(AIza[0-9A-Za-z-_]{35})\b"),
        "[REDACTED_GOOGLE_API_KEY]",
    ),
    (
        "GROK_XAI_KEY",
        re.compile(r"\b(xai-[a-zA-Z0-9]{30,})\b"),
        "[REDACTED_XAI_KEY]",
    ),
    # 5. JSON Web Tokens (JWT)
    (
        "JWT_TOKEN",
        re.compile(r"\b(ey[A-Za-z0-9_-]{10,}\.ey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"),
        "[REDACTED_JWT_TOKEN]",
    ),
    # 6. Bearer Authorization Headers
    (
        "BEARER_TOKEN",
        re.compile(r"(?i)\bBearer\s+([A-Za-z0-9_\-\.]{25,})\b"),
        "Bearer [REDACTED_BEARER_TOKEN]",
    ),
    # 7. Common Password / Secret variable assignments in code/configs
    (
        "ASSIGNED_SECRET",
        re.compile(
            r"""(?i)\b([a-zA-Z0-9_]*(?:password|passwd|pwd|secret|token|api[_-]?key|client[_-]?secret)[a-zA-Z0-9_]*)\s*([:=])\s*(['"])([^'"\r\n]{4,})\3"""
        ),
        r"\1 \2 \3[REDACTED_SECRET]\3",
    ),
]


def mask_secrets(text: str) -> tuple[str, list[dict]]:
    """Scan text and replace detected secrets with redaction labels.

    Args:
        text: Raw source code or context snippet.

    Returns:
        tuple of:
          - sanitized_text (str): text with secrets safely masked
          - detections (list of dict): metadata list of detected secrets
    """
    if not text:
        return text, []

    sanitized = text
    detections: list[dict] = []

    for name, pattern, replacement in SECRET_PATTERNS:
        matches = list(pattern.finditer(sanitized))
        if not matches:
            continue

        for m in matches:
            detections.append({
                "type": name,
                "matched_sample": m.group(0)[:15] + "...",
                "span": (m.start(), m.end()),
            })

        # Apply substitution
        sanitized = pattern.sub(replacement, sanitized)

    return sanitized, detections


def mask_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    """Apply masking to a collection of retrieved chunk objects.

    Each chunk's 'content' field will be sanitized.

    Args:
        chunks: List of chunk dictionaries containing 'content'.

    Returns:
        tuple of (sanitized_chunks, total_secrets_masked)
    """
    masked_chunks = []
    total_masked = 0

    for chunk in chunks:
        chunk_copy = dict(chunk)
        content = chunk_copy.get("content", "")
        sanitized_content, detections = mask_secrets(content)
        chunk_copy["content"] = sanitized_content
        if detections:
            total_masked += len(detections)
            chunk_copy["secrets_detected"] = len(detections)
        masked_chunks.append(chunk_copy)

    return masked_chunks, total_masked
