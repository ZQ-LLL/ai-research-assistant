"""
test_m3.py  —  M3 exit condition: chunks are embedded, stored, and retrieved.
Delete after M3 is confirmed.
"""

from utils.scraper import scrape_url
from utils.chunker import chunk_text
from utils.vectorstore import create_collection, add_chunks, query_chunks

URL = "https://www.ibm.com/think/news/ai-tech-trends-predictions-2026"
QUESTION = "What are the biggest AI trends in 2026?"

# --- Step 1: scrape and chunk ---
print("Scraping...")
text = scrape_url(URL)
chunks = chunk_text(text)
print(f"Got {len(chunks)} chunks from {URL}\n")

# --- Step 2: embed and store ---
print("Embedding and storing in ChromaDB...")
collection = create_collection()
add_chunks(collection, chunks, source_url=URL)
print(f"Stored {collection.count()} chunks in collection.\n")

# --- Step 3: retrieve ---
print(f"Querying: '{QUESTION}'\n")
results = query_chunks(collection, QUESTION, n_results=3)

for i, r in enumerate(results, 1):
    print(f"[{i}] distance={r['distance']}  source={r['source']}")
    print(f"     {r['text'][:200]}")
    print()
