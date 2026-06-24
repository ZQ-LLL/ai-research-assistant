"""
utils/reporter.py
Responsible for: generating a structured research report from retrieved chunks.

This is the "G" in RAG (Retrieval-Augmented Generation).
The chunks retrieved from ChromaDB act as the grounding context — Claude is
instructed to base every claim on these chunks and cite them inline.

Why does grounding matter?
Without it, the LLM would answer from its training data (possibly outdated,
unverifiable). By passing in the actual source text, we force every claim to
be traceable back to a real document, which is the core value of RAG over
plain chat.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "anthropic/claude-haiku-4-5"


def generate_report(question: str, chunks: list[dict]) -> str:
    """
    Generate a structured research report grounded in the retrieved chunks.

    chunks: list of {text, source, distance} dicts from query_chunks()

    Returns the report as a markdown string.
    """
    # Assign a citation number to each unique source URL
    sources: dict[str, int] = {}
    context_blocks = []

    for chunk in chunks:
        url = chunk["source"]
        if url not in sources:
            sources[url] = len(sources) + 1
        citation_num = sources[url]
        context_blocks.append(f"[Source {citation_num}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_blocks)
    reference_list = "\n".join(f"[{num}] {url}" for url, num in sources.items())

    prompt = f"""You are a research assistant. Write a structured research report that answers the question below.

STRICT RULES:
- Base every factual claim ONLY on the provided source excerpts.
- Cite sources inline using [1], [2], etc.
- If the sources don't contain enough information to answer something, say so explicitly — do not invent facts.

QUESTION:
{question}

SOURCE EXCERPTS:
{context}

FORMAT YOUR REPORT AS:
## Introduction
(1-2 sentences framing the topic)

## Key Findings
(bullet points, each with an inline citation)

## Conclusion
(2-3 sentences summarizing the answer)

## References
{reference_list}"""

    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content
