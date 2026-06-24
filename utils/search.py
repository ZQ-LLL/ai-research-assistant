"""
utils/search.py
Responsible for: web search via Tavily API.

Tavily is a search API built specifically for LLM agents. Unlike a regular
search engine, it returns clean text snippets rather than just URLs — so we
get usable content immediately without needing to scrape the page first.

The snippet (~200 chars) is useful for quick relevance checks; the full page
content comes later in utils/scraper.py when we need the complete text.
"""

import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web and return a list of results.

    Returns a list of dicts, each with:
      - url:     the page URL
      - title:   the page title
      - content: a short snippet (~200 chars) of the most relevant text
    """
    response = _client.search(query=query, max_results=max_results)

    return [
        {
            "url":     r["url"],
            "title":   r["title"],
            "content": r["content"],
        }
        for r in response["results"]
    ]
