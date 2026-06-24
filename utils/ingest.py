"""
utils/ingest.py
Responsible for: ingesting local files (PDF, CSV, Excel) into ChromaDB.

Why does this belong in the RAG pipeline?
The research agent currently only reads web pages. By ingesting local files
with the same chunk → embed → store flow, we let the agent answer questions
that combine external web research with the user's own documents — without
changing a single line of agent.py or vectorstore.py.
"""

import fitz  # pymupdf

from utils.chunker import chunk_text
from utils.vectorstore import add_chunks


def _extract_pdf_text(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF given its raw bytes.
    Joins pages with a newline so chunk boundaries don't fall mid-sentence
    across pages.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


def ingest_pdf(file_bytes: bytes, filename: str, collection) -> int:
    """
    Parse a PDF, chunk the text, and store chunks in ChromaDB.

    file_bytes: raw bytes from st.file_uploader (or open(path, "rb").read())
    filename:   used as the source label in metadata (shown in report citations)
    collection: ChromaDB collection to store into

    Returns the number of chunks stored.
    """
    text = _extract_pdf_text(file_bytes)
    if not text.strip():
        return 0

    chunks = chunk_text(text)
    # Use "file:<filename>" as source so citations are clearly
    # labelled as coming from an uploaded file, not a URL.
    add_chunks(collection, chunks, source_url=f"file:{filename}")
    return len(chunks)
