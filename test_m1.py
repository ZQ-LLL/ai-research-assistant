"""
test_m1.py  —  M1 exit condition: Tavily returns results for a test question.
Delete this file after M1 is confirmed working.
"""

from utils.search import search_web

query = "generative AI investment trends 2026"
print(f"Searching: '{query}'\n")

results = search_web(query, max_results=5)

print(f"Got {len(results)} results:\n")
for i, r in enumerate(results, 1):
    print(f"[{i}] {r['title']}")
    print(f"    {r['url']}")
    print(f"    {r['content'][:120]}...")
    print()
