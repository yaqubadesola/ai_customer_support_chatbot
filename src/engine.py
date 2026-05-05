"""
Chatbot engine for Meridian Electronics Customer Support.
Synchronous implementation using OpenAI sync client + sync MCP client.
"""

import json
import logging

from openai import OpenAI

from .auth import AuthSession
from .mcp_client import MCPClient, MCPError

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-4o-mini"

SYSTEM_PROMPT = """You are Yaqub, a friendly and efficient customer support agent for Meridian Electronics — a company that sells monitors, keyboards, printers, networking gear, and accessories.

## Your Capabilities (via tools)
You have access to MCP tools that connect to Meridian's live backend systems. Use them to:
- **Authenticate customers**: verify identity with email + PIN before accessing account data
- **Check product availability**: look up stock levels and product details
- **Look up order history**: retrieve orders for authenticated customers
- **Place orders**: help customers purchase products

## Authentication Rules (CRITICAL)
- NEVER access order history, account details, or place orders without a successfully authenticated session
- Authentication requires BOTH email AND a 4-digit PIN
- If a customer asks for account-related info without being authenticated, politely ask for their email and PIN first
- Once authenticated, you know who the customer is — use their name in responses
- If authentication fails, tell the customer and offer to try again (max 5 attempts)
- Sensitive operations require authentication; product browsing does NOT

## Communication Style
- Be warm, concise, and professional
- When showing order lists or product info, format it clearly (use markdown lists/tables)
- If a tool call fails, explain the issue plainly and offer alternatives
- Do not expose raw JSON or internal IDs unless the customer needs them
- When a customer is authenticated, acknowledge them by name

## Tool Use
- Always use the available tools to fetch real data — never make up product names, prices, or order details
- If a tool returns an error, surface a helpful message to the customer
- Chain tool calls logically (e.g., authenticate → then fetch orders)
"""


def _format_tool_result(result) -> str:
    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=2)
    return str(result)


class ChatEngine:
    def __init__(self, openrouter_api_key: str):
        self.client = OpenAI(
            api_key=openrouter_api_key,
            base_url=OPENROUTER_BASE_URL,
        )
        self.mcp = MCPClient()
        self.auth = AuthSession()
        self._tools_cache: list[dict] | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.mcp.initialize()
        self._tools_cache = self.mcp.get_openai_tools()
        self._initialized = True
        logger.info("ChatEngine initialized with %d tools", len(self._tools_cache))

    def _build_session_context(self) -> str:
        if self.auth.authenticated:
            return (
                f"\n\n## Current Session\n"
                f"Customer authenticated: {self.auth.customer_name} ({self.auth.customer_email})\n"
                f"Customer ID: {self.auth.customer_id}\n"
                f"You may access their account data freely."
            )
        return (
            "\n\n## Current Session\n"
            "Customer NOT authenticated yet.\n"
            f"Failed attempts this session: {self.auth.failed_attempts}\n"
            "Ask for email + PIN before accessing any account-specific data."
        )

    def _run_tool_loop(self, messages: list[dict]) -> tuple[str, list[dict]]:
        self._ensure_initialized()
        max_iterations = 8

        for _ in range(max_iterations):
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=self._tools_cache or [],
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1024,
            )

            choice = response.choices[0]
            assistant_msg = choice.message
            messages.append(assistant_msg.model_dump(exclude_unset=True))

            if not assistant_msg.tool_calls:
                return assistant_msg.content or "", messages

            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                logger.info("LLM calling tool: %s(%s)", tool_name, arguments)
                tool_result = self._invoke_tool_with_auth(tool_name, arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _format_tool_result(tool_result),
                })

        return "I'm sorry, I ran into an issue processing your request. Please try again.", messages

    def _invoke_tool_with_auth(self, tool_name: str, arguments: dict):
        try:
            result = self.mcp.call_tool(tool_name, arguments)
        except MCPError as e:
            logger.warning("Tool %s failed: %s", tool_name, e)
            return {"error": str(e)}

        if any(name in tool_name.lower() for name in ["auth", "login", "verify"]):
            self._handle_auth_result(arguments, result)

        return result

    def _handle_auth_result(self, arguments: dict, result) -> None:
        if isinstance(result, dict):
            if result.get("authenticated") or result.get("success") or result.get("customer"):
                customer_data = result.get("customer") or result
                self.auth.login(arguments.get("email", ""), customer_data)
                logger.info("Customer authenticated: %s", arguments.get("email"))
            elif result.get("error") or result.get("authenticated") is False:
                self.auth.record_failed_attempt()
        elif result and not isinstance(result, str):
            self.auth.login(arguments.get("email", ""), {"name": arguments.get("email", "")})

    def chat(self, user_message: str, history: list[dict]) -> str:
        try:
            self._ensure_initialized()
        except Exception:
            logger.exception("MCP initialization failed")
            return "I'm having trouble connecting to the backend. Please try again."

        system = SYSTEM_PROMPT + self._build_session_context()
        messages: list[dict] = [{"role": "system", "content": system}]

        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_message})

        try:
            final_text, _ = self._run_tool_loop(messages)
            return final_text
        except Exception:
            logger.exception("Unexpected error in chat loop")
            return "I'm experiencing a technical issue right now. Please try again in a moment."

    def logout(self) -> str:
        name = self.auth.greeting_name()
        self.auth.logout()
        return f"You've been logged out. Goodbye, {name}!"
