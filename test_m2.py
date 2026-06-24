"""
test_m2.py  —  M2 exit condition: scrape a URL, chunk the text, print results.
Delete after M2 is confirmed.
"""

from utils.scraper import scrape_url
from utils.chunker import chunk_text

url = "https://www.ibm.com/think/news/ai-tech-trends-predictions-2026"

print(f"Scraping: {url}\n")
text = scrape_url(url)

if text is None:
    print("Scraping failed — trafilatura returned None.")
else:
    print(f"Extracted {len(text)} characters of text.")
    print(f"\nFirst 400 chars:\n{text[:400]}\n")

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks (chunk_size=400, overlap=80).\n")
    print(f"Chunk [0]:\n{chunks[0]}\n")
    print(f"Chunk [1] (notice the overlap with chunk [0]):\n{chunks[1]}\n")
