"""
utils/scraper.py
Responsible for: extracting clean text content from a URL.

We split fetching and parsing into two steps:
  1. requests.get()      — gives us full control over timeout and headers
  2. trafilatura.extract() — strips HTML noise, returns clean article text

Why not use trafilatura.fetch_url()?
trafilatura's internal fetch doesn't reliably respect timeout on all platforms
(especially Chinese sites with slow or unusual responses). Using requests
directly ensures we never hang longer than TIMEOUT_SECONDS.
"""

import trafilatura
import requests

TIMEOUT_SECONDS = 12

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def scrape_url(url: str) -> str | None:
    """
    Fetch a URL and extract its main text content.

    Returns the clean article text, or None if anything fails
    (timeout, HTTP error, bot detection, no extractable content).
    Callers should skip None results rather than crashing.
    """
    try:
        response = requests.get(
            url,
            headers=_HEADERS,
            timeout=TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        response.raise_for_status()
        # apparent_encoding uses charset-normalizer to detect the real encoding,
        # which is more reliable than trusting the Content-Type header —
        # especially for Chinese sites that often declare latin-1 but serve UTF-8 or GBK.
        response.encoding = response.apparent_encoding
        html = response.text
    except Exception:
        return None

    return trafilatura.extract(html)
