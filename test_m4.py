"""
test_m4.py  —  M4 exit condition: full pipeline runs end-to-end and produces
               a cited research report.
Delete after M4 is confirmed.
"""

import time

from utils.search import search_web
from utils.scraper import scrape_url
from utils.chunker import chunk_text
from utils.vectorstore import create_collection, add_chunks, query_chunks
from utils.reporter import generate_report

QUESTION = "What are the biggest AI trends shaping enterprise technology in 2026?"
N_SEARCH_RESULTS = 4   # how many URLs to fetch
N_RETRIEVE       = 6   # how many chunks to feed to Claude

print("=" * 60)
print(f"QUESTION: {QUESTION}")
print("=" * 60)

# ── Step 1: Search ────────────────────────────────────────────
print(f"\n[1/4] Searching the web...")
t0 = time.time()
search_results = search_web(QUESTION, max_results=N_SEARCH_RESULTS)
print(f"      Found {len(search_results)} URLs ({time.time()-t0:.1f}s)")

# ── Step 2: Scrape + Chunk ────────────────────────────────────
print(f"\n[2/4] Scraping and chunking...")
t0 = time.time()
collection = create_collection()
total_chunks = 0

for r in search_results:
    url   = r["url"]
    text  = scrape_url(url)
    if text is None:
        print(f"      SKIP (scrape failed): {url}")
        continue
    chunks = chunk_text(text)
    add_chunks(collection, chunks, source_url=url)
    total_chunks += len(chunks)
    print(f"      +{len(chunks):3d} chunks  {url}")

print(f"      Total: {total_chunks} chunks stored ({time.time()-t0:.1f}s)")

# ── Step 3: Retrieve ──────────────────────────────────────────
print(f"\n[3/4] Retrieving top-{N_RETRIEVE} relevant chunks...")
t0 = time.time()
retrieved = query_chunks(collection, QUESTION, n_results=N_RETRIEVE)
for i, r in enumerate(retrieved, 1):
    print(f"      [{i}] dist={r['distance']}  {r['source']}")
print(f"      ({time.time()-t0:.1f}s)")

# ── Step 4: Generate report ───────────────────────────────────
print(f"\n[4/4] Generating report with Claude...")
t0 = time.time()
report = generate_report(QUESTION, retrieved)
print(f"      ({time.time()-t0:.1f}s)\n")

print("=" * 60)
print("REPORT")
print("=" * 60)
print(report)
