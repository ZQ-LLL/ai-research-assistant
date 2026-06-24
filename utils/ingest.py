"""
utils/ingest.py
Responsible for: ingesting local files (PDF, CSV, Excel) into ChromaDB.

Why does this belong in the RAG pipeline?
The research agent currently only reads web pages. By ingesting local files
with the same chunk → embed → store flow, we let the agent answer questions
that combine external web research with the user's own documents — without
changing a single line of agent.py or vectorstore.py.
"""

import io

import fitz  # pymupdf
import pandas as pd

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


# ── CSV / Excel ───────────────────────────────────────────────

def _dataframe_to_text(df: pd.DataFrame, filename: str) -> str:
    """
    Convert a DataFrame into a text description suitable for RAG.

    We produce three sections so different queries hit the right chunks:
      1. Overview — column names, dtypes, row count
      2. Per-column stats — numeric: min/max/mean/median; categorical: top values
      3. Sample rows — first 20 rows as plain text

    This is more useful than chunking raw CSV rows because vector search
    finds meaning, not exact values. "What is the average funding?" will
    match the stats section even if those words don't appear in the raw data.
    """
    lines = []

    # ── Section 1: Overview ──
    lines.append(f"DATASET: {filename}")
    lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append(f"Columns: {', '.join(str(c) for c in df.columns)}")
    lines.append("")

    # ── Section 2: Per-column stats ──
    for col in df.columns:
        lines.append(f"--- Column: {col} (type: {df[col].dtype}) ---")
        missing = int(df[col].isna().sum())
        lines.append(f"Missing values: {missing} ({missing / len(df) * 100:.1f}%)")

        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            if len(s):
                lines.append(
                    f"Min: {s.min():.4g}  Max: {s.max():.4g}  "
                    f"Mean: {s.mean():.4g}  Median: {s.median():.4g}"
                )
        else:
            top = df[col].value_counts().head(10)
            top_str = "  |  ".join(f"{v}: {c}" for v, c in top.items())
            lines.append(f"Top values: {top_str}")
        lines.append("")

    # ── Section 3: Sample rows ──
    lines.append("--- Sample rows (first 20) ---")
    lines.append(df.head(20).to_string(index=False))

    return "\n".join(lines)


def ingest_csv(file_bytes: bytes, filename: str, collection) -> int:
    """
    Parse a CSV or Excel file, convert it to a text description, and store
    in ChromaDB so the agent can answer questions about the dataset's content.

    file_bytes: raw bytes from st.file_uploader
    filename:   used for source label and to detect file type (.csv vs .xlsx)
    collection: ChromaDB collection to store into

    Returns the number of chunks stored.
    """
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(buf)
    else:
        df = pd.read_csv(buf)

    text = _dataframe_to_text(df, filename)
    chunks = chunk_text(text)
    add_chunks(collection, chunks, source_url=f"file:{filename}")
    return len(chunks)
