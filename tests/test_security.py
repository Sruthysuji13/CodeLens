import pytest
from app.security import mask_secrets, mask_chunks


def test_mask_private_key():
    sample = """
    -----BEGIN RSA PRIVATE KEY-----
    MIIEowIBAAKCAQEA0Y3wZ1...dummy...
    -----END RSA PRIVATE KEY-----
    """
    masked, detections = mask_secrets(sample)
    assert "[REDACTED_PRIVATE_KEY]" in masked
    assert "BEGIN RSA PRIVATE KEY" not in masked
    assert len(detections) >= 1
    assert detections[0]["type"] == "PRIVATE_KEY"


def test_mask_db_connection_string():
    sample = "postgres_url = 'postgresql://admin:superSecretPass123@db.internal:5432/production'"
    masked, detections = mask_secrets(sample)
    assert "superSecretPass123" not in masked
    assert "[REDACTED_PASSWORD]" in masked
    assert len(detections) == 1
    assert detections[0]["type"] == "DB_CONNECTION_STRING"


def test_mask_api_keys():
    dummy_openai = "sk-" + "proj1234567890abcdef1234567890"
    dummy_github = "ghp_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    dummy_slack = "xoxb-" + "1234567890-1234567890123-abcdefghijklmnopqrstuv"
    sample = f"""
    openai_key = '{dummy_openai}'
    github_pat = '{dummy_github}'
    slack_key = '{dummy_slack}'
    """
    masked, detections = mask_secrets(sample)
    assert dummy_openai not in masked
    assert "[REDACTED_OPENAI_KEY]" in masked
    assert dummy_github not in masked
    assert "[REDACTED_GITHUB_TOKEN]" in masked
    assert dummy_slack not in masked
    assert "[REDACTED_SLACK_TOKEN]" in masked
    assert len(detections) >= 3


def test_mask_assigned_secret():
    sample = 'DATABASE_PASSWORD = "MySuperSecretPassword#2026"'
    masked, detections = mask_secrets(sample)
    assert "MySuperSecretPassword#2026" not in masked
    assert "[REDACTED_SECRET]" in masked
    assert len(detections) == 1


def test_mask_chunks():
    chunks = [
        {"id": "chunk-1", "content": 'secret_key = "abc123XYZsecret!!"'},
        {"id": "chunk-2", "content": 'def add(a, b):\n    return a + b'},
    ]
    masked_chunks, total_masked = mask_chunks(chunks)
    assert total_masked == 1
    assert "[REDACTED_SECRET]" in masked_chunks[0]["content"]
    assert "return a + b" in masked_chunks[1]["content"]


def test_clean_code_no_false_positives():
    clean_code = """
    class AuthService:
        def __init__(self, db_client):
            self.db = db_client
            
        def authenticate(self, username, user_hash):
            if not username or not user_hash:
                return False
            return self.db.find_user(username) is not None
    """
    masked, detections = mask_secrets(clean_code)
    assert masked == clean_code
    assert len(detections) == 0


def test_empty_string():
    masked, detections = mask_secrets("")
    assert masked == ""
    assert detections == []

