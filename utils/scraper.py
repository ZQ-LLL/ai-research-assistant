"""
utils/scraper.py
Responsible for: extracting clean text content from a URL.

trafilatura is a library designed specifically for this task. Given a URL,
it fetches the page and strips out navigation bars, ads, footers, HTML tags,
and other noise — leaving only the main article text.

Why not just use requests + BeautifulSoup?
Because most pages mix article text with menus, sidebars, and ads.
Separating them reliably with CSS selectors is fragile and site-specific.
trafilatura uses heuristics trained on thousands of real pages to do this
automatically, which is why it's the standard choice for RAG pipelines.
"""

import trafilatura


def scrape_url(url: str) -> str | None:
    """
    Fetch a URL and extract its main text content.

    Returns the clean article text, or None if extraction fails
    (e.g. paywall, bot detection, connection error, no main content found).
    Callers should skip None results rather than crashing.
    """
    html = trafilatura.fetch_url(url)
    if html is None:
        return None

    text = trafilatura.extract(html)
    return text  # may still be None if trafilatura finds no main content
