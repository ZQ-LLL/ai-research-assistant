# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```powershell
# Activate venv and launch (PowerShell)
.\venv\Scripts\streamlit run app.py

# bash / Git Bash
./venv/Scripts/streamlit run app.py
```

App runs at `http://localhost:8501`. Streamlit hot-reloads on file save; if it doesn't pick up changes, kill the process and restart (stale `.pyc` files can interfere — delete `utils/__pycache__/` if needed).

## Environment

Copy `.env` and fill in both keys before running:

```
OPENROUTER_API_KEY=...   # Claude via OpenRouter (OpenAI-compatible)
TAVILY_API_KEY=...       # Web search
```

## Architecture

The pipeline runs in this order on every research question:

```
(optional) Uploaded PDFs
  → ingest.py: ingest_pdf()        # pymupdf → chunk → ChromaDB (same pipeline as web)

User question
  → agent.py: run_agent()          # drives the loop
      → search.py: search_web()    # Tavily → list of {url, title, snippet}
      → scraper.py: scrape_url()   # trafilatura → clean article text
      → chunker.py: chunk_text()   # split into 400-char overlapping chunks
      → vectorstore.py: add_chunks()  # sentence-transformers embed → ChromaDB store
      → vectorstore.py: query_chunks() # cosine similarity retrieval (web + PDFs unified)
      → reporter.py: generate_report() # Claude synthesises cited report
```

PDF sources appear in citations as `file:<filename>` to distinguish them from URLs.

**agent.py** is the orchestrator. It exposes three tools to Claude (`search_web`, `scrape_and_store`, `generate_report`) and runs an OpenAI-format tool-calling loop. When Claude calls `generate_report`, the loop exits immediately and returns that result — no second API call.

**vectorstore.py** uses `chromadb.EphemeralClient` (in-memory, resets on process exit). Each call to `create_collection()` generates a UUID-suffixed name to avoid collisions across Streamlit reruns in the same process.

**reporter.py** uses `anthropic/claude-haiku-4-5` (fast, cheap). **agent.py** uses `anthropic/claude-sonnet-4-5` (stronger reasoning needed for multi-step planning). Both go through OpenRouter with the OpenAI SDK (`base_url="https://openrouter.ai/api/v1"`).

## Key constraints

- The agent is designed for **research synthesis**, not structured data collection. Questions requiring a complete enumeration ("list all X") will produce incomplete results.
- ChromaDB collection is **reset per question** — there is no persistence between sessions by design.
- `generate_report` is the agent's **exit signal**: the loop returns its output immediately. Never make it a non-final tool.
- The `on_step` callback in `run_agent()` is the only coupling between the agent and the UI — keep `utils/` free of Streamlit imports.
