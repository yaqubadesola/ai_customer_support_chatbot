---
title: Meridian Electronics Support Chat
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.6.0
app_file: app.py
pinned: false
license: mit
---

# ⚡ Meridian Electronics — AI Customer Support Chatbot

A production-ready AI support chatbot for Meridian Electronics built with:
- **Gradio** UI
- **OpenRouter** (GPT-4o-mini) LLM
- **MCP** (Model Context Protocol) backend integration

## Features
- 🔐 Customer authentication (email + 4-digit PIN)
- 📦 Order history lookup
- 🔍 Product availability checks  
- 🛒 Order placement
- 🤖 Agentic tool-use loop (LLM decides which tools to call)

## Architecture

```
User → Gradio UI → ChatEngine → OpenRouter (GPT-4o-mini)
                       ↕                    ↕
                  AuthSession         MCP Tool Loop
                       ↕                    ↕
                  Session State    MCP Server (GCP)
```

## Test Credentials

| Email | PIN |
|-------|-----|
| donaldgarcia@example.net | 7912 |
| michellejames@example.com | 1520 |
| laurahenderson@example.org | 1488 |
| spenceamanda@example.org | 2535 |
| glee@example.net | 4582 |

## Local Setup

```bash
git clone <repo>
cd meridian-chatbot
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY
python app.py
```

## Deployment

**HuggingFace Spaces**: Push repo, set `OPENROUTER_API_KEY` in Space secrets.

**Local**: Set `OPENROUTER_API_KEY` env var and run `python app.py`.
