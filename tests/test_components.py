"""
Basic unit tests for Meridian chatbot components.
Run: python -m pytest tests/ -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.auth import CustomerSession
from src.mcp_client import MCPClient


# ---------------------------------------------------------------------------
# CustomerSession tests
# ---------------------------------------------------------------------------

class TestCustomerSession:
    def test_initial_state_not_authenticated(self):
        s = CustomerSession()
        assert s.authenticated is False
        assert s.email is None
        assert s.customer_id is None

    def test_login_sets_auth(self):
        s = CustomerSession()
        s.login(email="test@example.com", customer_id="42", name="Jane")
        assert s.authenticated is True
        assert s.email == "test@example.com"
        assert s.customer_id == "42"
        assert s.customer_name == "Jane"

    def test_logout_resets_state(self):
        s = CustomerSession()
        s.login("a@b.com", "1", "Alice")
        s.logout()
        assert s.authenticated is False
        assert s.email is None

    def test_display_name_fallback(self):
        s = CustomerSession()
        s.login("a@b.com", "1")
        assert s.display_name == "a@b.com"

    def test_display_name_prefers_name(self):
        s = CustomerSession()
        s.login("a@b.com", "1", name="Alice")
        assert s.display_name == "Alice"

    def test_context_string_unauthenticated(self):
        s = CustomerSession()
        ctx = s.to_context_string()
        assert "NO AUTHENTICATED" in ctx
        assert "authenticate" in ctx.lower()

    def test_context_string_authenticated(self):
        s = CustomerSession()
        s.login("a@b.com", "99", "Alice")
        ctx = s.to_context_string()
        assert "AUTHENTICATED" in ctx
        assert "a@b.com" in ctx
        assert "99" in ctx


# ---------------------------------------------------------------------------
# MCPClient tests
# ---------------------------------------------------------------------------

class TestMCPClient:
    def test_get_openai_schemas_empty_when_no_tools(self):
        client = MCPClient()
        client._tools = []
        schemas = client.get_openai_tool_schemas()
        assert schemas == []

    def test_get_openai_schemas_format(self):
        client = MCPClient()
        client._tools = [
            {
                "name": "check_product",
                "description": "Check product availability",
                "inputSchema": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                },
            }
        ]
        schemas = client.get_openai_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "check_product"
        assert "sku" in schemas[0]["function"]["parameters"]["properties"]

    def test_parse_response_json(self):
        client = MCPClient()
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {"result": {"tools": []}}
        result = client._parse_response(mock_resp)
        assert result == {"result": {"tools": []}}

    def test_parse_response_sse(self):
        client = MCPClient()
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "text/event-stream"}
        mock_resp.text = 'data: {"result": {"tools": []}}\n\n'
        result = client._parse_response(mock_resp)
        assert result == {"result": {"tools": []}}

    def test_call_tool_handles_timeout(self):
        import httpx
        client = MCPClient()
        client._tools = []

        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            result = client.call_tool("some_tool", {})

        assert "error" in result
        assert "timed out" in result["error"].lower()

    def test_call_tool_extracts_text_content(self):
        client = MCPClient()
        mock_resp = MagicMock()
        mock_resp.headers = {"content-type": "application/json"}
        mock_resp.json.return_value = {
            "result": {
                "content": [{"type": "text", "text": '{"status": "ok", "stock": 5}'}]
            }
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_resp):
            result = client.call_tool("get_stock", {"sku": "MON-001"})

        assert result == {"status": "ok", "stock": 5}
