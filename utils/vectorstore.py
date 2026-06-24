"""
utils/vectorstore.py
Responsible for: embedding text chunks and storing/querying them in ChromaDB.

Two key concepts:

1. Embeddings
   A text chunk like "GPT-4 was released in 2023" gets converted into a list
   of ~384 numbers (a vector). Similar sentences produce similar vectors.
   sentence-transformers does this conversion using a small neural network
   (all-MiniLM-L6-v2, ~90MB) that runs locally — no API call needed.

2. ChromaDB
   An in-memory vector database. We store each chunk alongside its embedding
   and a metadata dict (e.g. {"source": url}). When we query with a question,
   ChromaDB embeds the question and finds the stored chunks whose vectors are
   closest — i.e. most semantically similar — using cosine similarity.

Why not just search for keywords?
Because "LLM inference speed" and "how fast language models run" have zero
words in common but are semantically identical. Vector search captures meaning,
not just word overlap.
"""

import uuid

import chromadb
from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_model = SentenceTransformer(_EMBEDDING_MODEL)


def create_collection() -> chromadb.Collection:
    """
    Create a fresh in-memory ChromaDB collection for one research session.
    EphemeralClient means everything is reset when the process ends — no files
    are written to disk. Call this once at the start of each research question.
    """
    client = chromadb.EphemeralClient()
    # Use a unique name so there is never a collision, even if EphemeralClient
    # shares in-process state across Streamlit reruns.
    name = f"research_{uuid.uuid4().hex}"
    return client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # cosine distance: 0=identical, 1=unrelated
    )


def add_chunks(collection: chromadb.Collection, chunks: list[str], source_url: str) -> None:
    """
    Embed a list of text chunks and add them to the collection.
    Each chunk is stored with its embedding and the source URL as metadata.
    """
    if not chunks:
        return

    embeddings = _model.encode(chunks).tolist()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source_url} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )


def query_chunks(
    collection: chromadb.Collection,
    question: str,
    n_results: int = 5,
) -> list[dict]:
    """
    Find the n_results chunks most relevant to question.

    Returns a list of dicts, each with:
      - text:     the chunk content
      - source:   the URL it came from
      - distance: cosine distance (lower = more relevant, 0 is perfect match)
    """
    query_embedding = _model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    return [
        {
            "text":     doc,
            "source":   meta["source"],
            "distance": round(dist, 4),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
