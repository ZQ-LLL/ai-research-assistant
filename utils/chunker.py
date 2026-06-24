"""
utils/chunker.py
Responsible for: splitting a long text into overlapping chunks.

Why chunk at all?
LLMs have a context window limit, so we can't feed them an entire article.
More importantly, vector search works best on short, focused pieces of text —
a 5000-word article has many different topics mixed together, but a 400-char
chunk usually contains just one idea, which makes retrieval more precise.

Why overlap?
A sentence near the boundary of a chunk would get cut in half without overlap.
Overlap ensures that every sentence appears fully in at least one chunk.
"""


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """
    Split text into overlapping chunks of approximately chunk_size characters.

    chunk_size: target length of each chunk in characters (~3-4 sentences)
    overlap:    how many characters the next chunk repeats from the previous one
                (prevents information loss at chunk boundaries)

    Returns a list of non-empty strings.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # step forward, but back-track by overlap

    return chunks
