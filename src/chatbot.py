"""
Chatbot engine for Meridian Electronics Customer Support.
Orchestrates the OpenRouter LLM (GPT-4o) with MCP tool calls.
"""

import json
import logging
import os
from typing import Generator

from openai import OpenAI

from src.mcp_client import MCPClient
from src.auth import CustomerSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenRouter client (drop-in OpenAI-compatible API)
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", "openai/gpt-4o")          # cost-effective default

openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://meridian-electronics.app",
        "X-Title": "Meridian Customer Support",
    },
)

# ---------------------------------------------------------------------------
# System prompt factory
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """You are Aria, the friendly and efficient AI customer support agent for Meridian Electronics — a company that sells monitors, keyboards, printers, networking gear, and accessories.

Your personality:
- Warm, professional, and concise
- Proactively helpful — anticipate follow-up needs
- Honest about limitations; never fabricate order or product data

Your capabilities (via tools):
- Authenticate returning customers (email + PIN)
- Check product availability / catalogue
- Look up order history and order status
- Place new orders on behalf of authenticated customers

AUTHENTICATION RULES — follow these strictly:
1. Any request involving a specific account (orders, purchase history, placing an order) requires authentication FIRST.
2. Collect the customer's email, then their PIN, then call authenticate_customer.
3. Once authenticated, greet them by name and proceed with their request.
4. Product availability/catalogue questions do NOT require authentication.
5. Never reveal PIN numbers or raw customer IDs in your replies.

TOOL USAGE RULES:
- Always call the real MCP tools — never make up order numbers, prices, or stock levels.
- If a tool returns an error, explain it clearly to the customer and offer alternatives.
- After placing an order, confirm the order ID and summary to the customer.

Session context:
{session_context}
"""


def build_system_prompt(session: CustomerSession) -> str:
    return SYSTEM_TEMPLATE.format(session_context=session.to_context_string())


# ---------------------------------------------------------------------------
# Agentic loop (streaming-friendly, handles tool calls)
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 6  # guard against infinite loops


def chat(
    user_message: str,
    history: list[dict],        # OpenAI-format message list
    session: CustomerSession,
    mcp: MCPClient,
) -> Generator[str, None, None]:
    """
    Run one user turn through the agentic loop.
    Yields incremental text chunks for Gradio streaming.
    Updates `session` in-place when authentication succeeds.
    Returns the final assistant message via the last yield.
    """
    tools = mcp.get_openai_tool_schemas()

    messages = [{"role": "system", "content": build_system_prompt(session)}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    full_reply = ""

    for _round in range(MAX_TOOL_ROUNDS):
        response = openrouter.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            stream=False,   # process tool calls cleanly; stream text on final turn
        )

        choice = response.choices[0]
        msg = choice.message

        # ---- Pure text reply (no tool calls) → stream it out ----
        if not msg.tool_calls:
            full_reply = msg.content or ""
            yield full_reply
            return

        # ---- Tool call(s) requested ----
        # Append assistant message with tool_calls to history
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each tool call
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            logger.info("Tool call: %s(%s)", tool_name, arguments)

            # Yield a lightweight status so the UI doesn't look frozen
            yield f"_⚙️ Calling **{tool_name}**…_\n\n"

            result = mcp.call_tool(tool_name, arguments)

            # ---- Update session state after successful auth ----
            if tool_name == "authenticate_customer" and isinstance(result, dict):
                if result.get("authenticated") or result.get("success"):
                    session.login(
                        email=arguments.get("email", ""),
                        customer_id=str(result.get("customer_id", result.get("id", ""))),
                        name=result.get("name", result.get("customer_name", "")),
                    )
                    logger.info("Customer authenticated: %s", session.email)

            # Append tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

        # Update system prompt with fresh auth state for next round
        messages[0]["content"] = build_system_prompt(session)

    # Safety fallback — shouldn't normally reach here
    yield "I'm sorry, I ran into an issue processing your request. Please try again."
