"""
Meridian Electronics — AI Customer Support Chatbot
Gradio UI + OpenRouter (GPT-4o-mini) + MCP backend
"""

import logging
import os
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Add src to path when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.engine import ChatEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom CSS — clean dark-tech look suited to a B2C electronics brand
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
* { box-sizing: border-box; font-family: system-ui, sans-serif; }

.gradio-container {
  background: #ffffff !important;
  max-width: 800px !important;
  margin: 0 auto !important;
  padding: 24px !important;
}

.meridian-header {
  padding: 16px 0 12px;
  border-bottom: 1px solid #e5e7eb;
  margin-bottom: 16px;
}
.meridian-logo {
  font-size: 18px;
  font-weight: 600;
  color: #111827;
}
.meridian-logo span { color: #2563eb; }
.meridian-tagline {
  font-size: 12px;
  color: #9ca3af;
  margin-top: 2px;
}

.auth-pill {
  font-size: 12px;
  color: #6b7280;
  padding: 4px 0;
}
.auth-pill.authed { color: #16a34a; }

#chatbot {
  background: #ffffff !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 8px !important;
}

#msg-input textarea {
  border: 1px solid #e5e7eb !important;
  border-radius: 6px !important;
  font-size: 14px !important;
  resize: none !important;
  background: #ffffff !important;
  color: #111827 !important;
}
#msg-input textarea:focus {
  border-color: #2563eb !important;
  outline: none !important;
  box-shadow: none !important;
}

#send-btn {
  background: #2563eb !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 6px !important;
  font-size: 14px !important;
  font-weight: 500 !important;
}
#send-btn:hover { background: #1d4ed8 !important; }

#logout-btn, #clear-btn {
  background: transparent !important;
  border: 1px solid #e5e7eb !important;
  color: #6b7280 !important;
  border-radius: 6px !important;
  font-size: 12px !important;
}
#logout-btn:hover { color: #ef4444 !important; border-color: #ef4444 !important; }
#clear-btn:hover { color: #374151 !important; }

.chip-btn {
  background: #f9fafb !important;
  border: 1px solid #e5e7eb !important;
  color: #374151 !important;
  border-radius: 6px !important;
  font-size: 12px !important;
}
.chip-btn:hover {
  border-color: #2563eb !important;
  color: #2563eb !important;
  background: #eff6ff !important;
}

.info-panel {
  font-size: 12px;
  color: #6b7280;
  padding: 10px 0;
  border-top: 1px solid #e5e7eb;
  margin-top: 8px;
}
.info-panel code {
  background: #f3f4f6;
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 11px;
  color: #374151;
}

footer, .footer { display: none !important; }
"""

# ---------------------------------------------------------------------------
# Quick-access suggestion prompts
# ---------------------------------------------------------------------------
SUGGESTIONS_GUEST = [
    "Show available monitors",
    "What keyboards are in stock?",
    "Check printer availability",
    "What networking gear is available?",
]
SUGGESTIONS_AUTHED = [
    "Show my recent orders",
    "Track my latest order",
    "What networking gear is available?",
    "I'd like to place an order",
]

WELCOME_MESSAGE = (
    "Hi! I'm **Yaqub**, Meridian Electronics' support agent.\n\n"
    "I can help you browse products, check stock, track orders, and place new ones. "
    "For orders, just tell me your email and PIN when prompted.\n\n"
    "What can I help you with?"
)


# ---------------------------------------------------------------------------
# Session engine store (one ChatEngine per Gradio session)
# ---------------------------------------------------------------------------
_engines: dict[str, ChatEngine] = {}


def _get_engine(session_state: dict) -> ChatEngine | None:
    key = session_state.get("engine_key")
    return _engines.get(key) if key else None


def _create_engine(session_state: dict, api_key: str) -> ChatEngine:
    import uuid
    key = str(uuid.uuid4())
    engine = ChatEngine(api_key)
    _engines[key] = engine
    session_state["engine_key"] = key
    return engine


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------
def on_load(session_state: dict):
    """Auto-initialise engine from environment key on page load."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if api_key:
        _create_engine(session_state, api_key)
    return session_state


def on_message(user_msg: str, history: list, session_state: dict):
    if not user_msg.strip():
        return history, history, session_state

    engine = _get_engine(session_state)
    if engine is None:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if api_key:
            engine = _create_engine(session_state, api_key)
        else:
            history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": "⚠️ No API key configured."},
            ]
            return history, history, session_state

    response = engine.chat(user_msg, history)
    history = history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": response},
    ]
    return history, history, session_state


def on_suggestion_click(chip: str, history: list, session_state: dict):
    """When a suggestion chip is clicked, return it as the input."""
    return chip


def on_logout(history: list, session_state: dict):
    engine = _get_engine(session_state)
    if engine:
        msg = engine.logout()
        history = history + [{"role": "assistant", "content": msg}]
    return history, history, session_state


def on_clear(session_state: dict):
    """Clear chat history."""
    engine = _get_engine(session_state)
    if engine:
        engine.auth.logout()
    return [], []


def get_auth_status_html(session_state: dict) -> str:
    engine = _get_engine(session_state)
    if engine and engine.auth.authenticated:
        return f'<div class="auth-pill authed">Logged in as {engine.auth.customer_name}</div>'
    return '<div class="auth-pill">Guest session</div>'


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="Meridian Electronics — Support Chat") as demo:
        session_state = gr.State({})

        # ── Header ──────────────────────────────────────────────────
        gr.HTML("""
        <div class="meridian-header">
          <div class="meridian-logo">Meridian<span>.</span>Electronics</div>
          <div class="meridian-tagline">Customer Support</div>
        </div>
        """)

        # ── Chat Panel ───────────────────────────────────────────────
        with gr.Column(visible=True) as chat_panel:
            with gr.Row():
                auth_status = gr.HTML('<div class="auth-pill">Guest session</div>')
                logout_btn = gr.Button("Logout", elem_id="logout-btn", scale=0)
                clear_btn = gr.Button("🗑 Clear", elem_id="clear-btn", scale=0)

            chatbot_kwargs = dict(
                value=[{"role": "assistant", "content": WELCOME_MESSAGE}],
                elem_id="chatbot",
                label="",
                show_label=False,
                render_markdown=True,
                height=480,
            )
            import gradio as _gr
            if tuple(int(x) for x in _gr.__version__.split(".")[:2]) < (6, 0):
                chatbot_kwargs["type"] = "messages"
            chatbot = gr.Chatbot(**chatbot_kwargs)

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your message…  (Enter to send)",
                    show_label=False,
                    scale=5,
                    lines=1,
                    max_lines=4,
                    elem_id="msg-input",
                )
                send_btn = gr.Button("Send ↵", variant="primary", scale=1, elem_id="send-btn")

            # Suggestion chips — defined after msg_input so they can target it
            with gr.Row(elem_classes="suggestion-row"):
                chips = []
                for s in SUGGESTIONS_GUEST:
                    chip = gr.Button(s, elem_classes="chip-btn", size="sm")
                    chips.append((chip, s))

            gr.HTML("""
            <div class="info-panel">
              Test: <code>donaldgarcia@example.net</code> / PIN <code>7912</code>
              &nbsp;&nbsp;|&nbsp;&nbsp; <code>michellejames@example.com</code> / PIN <code>1520</code>
            </div>
            """)

        # ── Chat history state ───────────────────────────────────────
        history_state = gr.State([{"role": "assistant", "content": WELCOME_MESSAGE}])

        # ── Wire events ─────────────────────────────────────────────
        demo.load(
            fn=on_load,
            inputs=[session_state],
            outputs=[session_state],
        )

        send_btn.click(
            fn=on_message,
            inputs=[msg_input, history_state, session_state],
            outputs=[chatbot, history_state, session_state],
        )

        msg_input.submit(
            fn=on_message,
            inputs=[msg_input, history_state, session_state],
            outputs=[chatbot, history_state, session_state],
        )

        logout_btn.click(
            fn=on_logout,
            inputs=[history_state, session_state],
            outputs=[chatbot, history_state, session_state],
        )

        clear_btn.click(
            fn=on_clear,
            inputs=[session_state],
            outputs=[chatbot, history_state],
        )

        # Wire suggestion chips: send chip text as message directly
        for chip, text in chips:
            chip.click(
                fn=lambda t=text: t,
                outputs=msg_input,
            ).then(
                fn=on_message,
                inputs=[msg_input, history_state, session_state],
                outputs=[chatbot, history_state, session_state],
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_error=True,
        css=CUSTOM_CSS,
    )
