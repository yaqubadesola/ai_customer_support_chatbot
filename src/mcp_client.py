"""
MCP Client for Meridian Electronics Support Chatbot.
Handles tool discovery and invocation via Streamable HTTP transport.
"""

import json
import logging
import httpx
from typing import Any

logger = logging.getLogger(__name__)

MCP_SERVER_URL = "https://order-mcp-74afyau24q-uc.a.run.app/mcp"
TIMEOUT = 30.0


class MCPError(Exception):
    pass


class MCPClient:
    def __init__(self, server_url: str = MCP_SERVER_URL):
        self.server_url = server_url
        self._tools: list[dict] | None = None
        self._session_id: str | None = None
        self._client = httpx.Client(timeout=TIMEOUT)

    def _rpc(self, method: str, params: dict | None = None) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        try:
            response = self._client.post(self.server_url, json=payload, headers=headers)
            response.raise_for_status()

            if "Mcp-Session-Id" in response.headers:
                self._session_id = response.headers["Mcp-Session-Id"]

            content_type = response.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                return self._parse_sse(response.text)

            data = response.json()
            if "error" in data:
                raise MCPError(f"MCP error {data['error'].get('code')}: {data['error'].get('message')}")
            return data.get("result")

        except httpx.HTTPStatusError as e:
            raise MCPError(f"HTTP {e.response.status_code}: {e.response.text[:200]}") from e
        except httpx.RequestError as e:
            raise MCPError(f"Request failed: {e}") from e

    def _parse_sse(self, raw: str) -> Any:
        result = None
        for line in raw.splitlines():
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "result" in data:
                        result = data["result"]
                    elif "error" in data:
                        raise MCPError(f"MCP SSE error: {data['error'].get('message')}")
                except json.JSONDecodeError:
                    pass
        return result

    def initialize(self) -> None:
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "meridian-chatbot", "version": "1.0.0"},
            },
        )
        logger.info("MCP session initialised (session=%s)", self._session_id)

    def list_tools(self) -> list[dict]:
        if self._tools is not None:
            return self._tools
        result = self._rpc("tools/list")
        tools = result.get("tools", []) if result else []
        self._tools = tools
        logger.info("Discovered %d tools: %s", len(tools), [t["name"] for t in tools])
        return tools

    def call_tool(self, name: str, arguments: dict) -> Any:
        logger.debug("Calling tool %s with %s", name, arguments)
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        if result is None:
            return None
        content = result.get("content", [])
        parts = [block["text"] for block in content if block.get("type") == "text"]
        raw = "\n".join(parts)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw

    def get_openai_tools(self) -> list[dict]:
        tools = self.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
                },
            }
            for tool in tools
        ]

    def close(self) -> None:
        self._client.close()
