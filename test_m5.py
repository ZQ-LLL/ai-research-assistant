"""
test_m5.py  —  M5 exit condition: agent autonomously searches, scrapes,
               and produces a cited report with no hardcoded query sequence.
Delete after M5 is confirmed.
"""

from utils.vectorstore import create_collection
from utils.agent import run_agent

QUESTION = "What are the biggest AI trends shaping enterprise technology in 2026?"

collection = create_collection()


def print_step(event: dict):
    """Print each agent step so we can watch it reason."""
    if event["type"] == "tool_call":
        name = event["name"]
        args = event["args"]
        if name == "search_web":
            print(f"\n>>> SEARCH: \"{args['query']}\"")
        elif name == "scrape_and_store":
            print(f"    SCRAPE: {args['url']}")
        elif name == "generate_report":
            print(f"\n>>> GENERATE REPORT (retrieving {args.get('n_results', 8)} chunks)...")

    elif event["type"] == "tool_result":
        name = event["name"]
        result = event["result"]
        if name == "scrape_and_store":
            print(f"           → {result}")
        elif name == "search_web":
            # Just show how many results came back
            n = result.count("URL:")
            print(f"           → {n} results returned")

    elif event["type"] == "limit":
        print("\n[!] Agent hit the step limit.")


print(f"QUESTION: {QUESTION}\n")
print("=" * 60)

report = run_agent(QUESTION, collection, on_step=print_step)

print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)
print(report)
