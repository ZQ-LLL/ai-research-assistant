"""
utils/agent.py
Responsible for: the Agent loop — tool definitions, tool execution, and the
                 main loop that drives Claude to research a question autonomously.

How the agent works:
  1. We give Claude a system prompt explaining its role and the workflow.
  2. We give Claude three tools it can call: search_web, scrape_and_store,
     generate_report.
  3. Claude decides — step by step — which tool to call and with what arguments.
  4. We execute the tool, return the result to Claude, and repeat.
  5. When Claude calls generate_report, that IS the final answer — we return it
     immediately without making another API call.

This "loop until done" pattern is the core of every LLM agent. The key insight
is that the model doesn't just run once — it can take multiple actions, see the
results, and adjust its plan before committing to a final answer.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from utils.chunker import chunk_text
from utils.reporter import generate_report as _generate_report
from utils.scraper import scrape_url
from utils.search import search_web as _search_web
from utils.vectorstore import add_chunks, query_chunks

load_dotenv()

_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=90.0,   # hard limit per API call; prevents silent hangs
)

MODEL = "anthropic/claude-sonnet-4-5"

SYSTEM_PROMPT = """You are a research assistant with access to web search and scraping tools.

Your goal: produce a well-cited, factual research report that answers the user's question.

Workflow:
1. Rephrase the question into 1-3 focused search queries (more specific = better results).
2. Call search_web for each query. Review the returned titles and snippets.
3. Call scrape_and_store for the 2-4 most relevant URLs.
4. If important aspects of the question are not yet covered, run another search.
5. Once you have scraped at least 3 good sources, call generate_report to finish.

Rules:
- Never answer from your own knowledge — use only what the tools return.
- Be selective: only scrape pages whose snippets suggest they directly answer the question.
- Scrape at most 5 URLs total. Stop scraping as soon as you have 3 good sources.
- Call generate_report exactly once, as your final action.
- Do not summarize or comment after generate_report — it IS your final response."""

# ── Tool schemas (OpenAI function-calling format) ─────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for pages relevant to a query. "
                "Returns a list of results with URL, title, and a short snippet. "
                "Use the snippet to decide which URLs are worth scraping in full. "
                "Call this multiple times with different queries to cover different angles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query. Rephrase the user's question into specific terms.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return. Default 4, max 6.",
                        "default": 4,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_and_store",
            "description": (
                "Fetch the full text of a web page and store it in the research database. "
                "Call this for URLs that look highly relevant based on their title and snippet. "
                "Returns the number of text chunks stored, or an error message if scraping fails."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to scrape and store.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "Retrieve the most relevant content from the research database and generate "
                "a structured, cited report. Call this ONLY when you have scraped enough sources "
                "(at least 3) to comprehensively answer the question. This is always your last action."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The original research question from the user.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve for the report. Default 8.",
                        "default": 8,
                    },
                },
                "required": ["question"],
            },
        },
    },
]


# ── Tool execution ────────────────────────────────────────────

def _execute_tool(name: str, args: dict, collection) -> str:
    """Run a tool and return its result as a string for the model to read."""

    if name == "search_web":
        results = _search_web(args["query"], max_results=args.get("max_results", 4))
        lines = []
        for r in results:
            lines.append(f"Title:   {r['title']}\nURL:     {r['url']}\nSnippet: {r['content']}\n")
        return "\n".join(lines) if lines else "No results found."

    elif name == "scrape_and_store":
        url = args["url"]
        text = scrape_url(url)
        if text is None:
            return f"Failed to scrape {url} — page may be paywalled or bot-protected."
        chunks = chunk_text(text)[:60]  # cap per-page to keep embedding time bounded
        add_chunks(collection, chunks, source_url=url)
        return f"Stored {len(chunks)} chunks from: {url}"

    elif name == "generate_report":
        question = args["question"]
        n = args.get("n_results", 8)
        retrieved = query_chunks(collection, question, n_results=n)
        if not retrieved:
            return "No relevant content in the database yet. Scrape more sources first."
        return _generate_report(question, retrieved)

    else:
        return f"Unknown tool: {name}"


# ── Agent loop ────────────────────────────────────────────────

def run_agent(
    question: str,
    collection,
    max_steps: int = 15,
    on_step=None,
) -> str:
    """
    Run the research agent loop.

    question:   the user's research question
    collection: a ChromaDB collection (created fresh per session)
    max_steps:  hard limit on tool-call rounds to prevent infinite loops
    on_step:    optional callback(event: dict) for UI progress updates.
                event shapes:
                  {"type": "tool_call",   "name": str, "args": dict}
                  {"type": "tool_result", "name": str, "result": str}
                  {"type": "limit"}

    Returns the final report as a markdown string.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    print(f"[agent] starting — question: {question[:80]}")

    for step in range(max_steps):
        print(f"[agent] step {step + 1}/{max_steps} — calling API...")
        response = _client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        print(f"[agent] step {step + 1} — API returned")

        msg = response.choices[0].message

        # No tool calls = agent decided to reply directly (shouldn't happen with
        # our prompt, but handle gracefully)
        if not msg.tool_calls:
            return msg.content or "Agent finished without generating a report."

        # Add the assistant turn (with tool_calls) to history
        messages.append({
            "role":       "assistant",
            "content":    msg.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # Execute each tool call and collect results
        final_report = None

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            if on_step:
                on_step({"type": "tool_call", "name": name, "args": args})

            result = _execute_tool(name, args, collection)

            if on_step:
                on_step({"type": "tool_result", "name": name, "result": result})

            # generate_report is the exit signal — capture its output
            if name == "generate_report":
                final_report = result

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

        if final_report is not None:
            return final_report

    # Safety net: hit max_steps without a report
    if on_step:
        on_step({"type": "limit"})
    return "Research agent reached the step limit without completing the report."
