"""
Unit tests for Meridian chatbot core logic.
Run with: python -m pytest tests/ -v
"""

import pytest
from src.auth import AuthSession


class TestAuthSession:

    def test_initial_state(self):
        session = AuthSession()
        assert session.authenticated is False
        assert session.customer_email is None
        assert session.failed_attempts == 0
        assert session.is_locked_out is False

    def test_login(self):
        session = AuthSession()
        session.login("test@example.com", {"name": "Test User", "id": "cust_123"})
        assert session.authenticated is True
        assert session.customer_email == "test@example.com"
        assert session.customer_name == "Test User"
        assert session.customer_id == "cust_123"

    def test_logout(self):
        session = AuthSession()
        session.login("test@example.com", {"name": "Test User"})
        session.logout()
        assert session.authenticated is False
        assert session.customer_email is None

    def test_failed_attempts(self):
        session = AuthSession()
        for _ in range(4):
            session.record_failed_attempt()
        assert session.is_locked_out is False
        session.record_failed_attempt()
        assert session.is_locked_out is True

    def test_failed_attempts_reset_on_login(self):
        session = AuthSession()
        session.record_failed_attempt()
        session.record_failed_attempt()
        session.login("x@x.com", {"name": "X"})
        assert session.failed_attempts == 0

    def test_greeting_name_with_name(self):
        session = AuthSession()
        session.login("a@b.com", {"name": "Alice"})
        assert session.greeting_name() == "Alice"

    def test_greeting_name_fallback(self):
        session = AuthSession()
        assert session.greeting_name() == "there"


class TestMCPClientParsing:
    """Test SSE parsing logic without network calls."""

    def test_parse_sse_happy_path(self):
        from src.mcp_client import MCPClient
        client = MCPClient()
        raw = 'data: {"result": {"tools": [{"name": "auth"}]}}\n\n'
        result = client._parse_sse(raw)
        assert result == {"tools": [{"name": "auth"}]}

    def test_parse_sse_empty(self):
        from src.mcp_client import MCPClient
        client = MCPClient()
        result = client._parse_sse("")
        assert result is None

    def test_parse_sse_error(self):
        from src.mcp_client import MCPClient, MCPError
        client = MCPClient()
        raw = 'data: {"error": {"code": -32000, "message": "Not found"}}\n'
        with pytest.raises(MCPError):
            client._parse_sse(raw)
