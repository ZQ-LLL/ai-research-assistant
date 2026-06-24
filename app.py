"""
app.py
Streamlit UI for the AI Research Assistant.

Flow:
  1. User types a research question and clicks "Research".
  2. A fresh ChromaDB collection is created for this session.
  3. The agent runs in a background thread so the main thread stays alive
     and keeps the WebSocket connection open with the browser.
  4. The main thread polls a shared event list and updates the status widget.
  5. When the agent finishes, the report is stored in session state and rendered.
"""

import threading
import time

import streamlit as st

from utils.agent import run_agent
from utils.ingest import ingest_csv, ingest_pdf
from utils.vectorstore import create_collection

# ── Page config ───────────────────────────────────────────────

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────

st.title("🔬 AI Research Assistant")
st.caption(
    "Searches the web, reads sources, and generates a cited research report. "
    "Powered by Tavily search + ChromaDB + Claude."
)
st.divider()

# ── File upload (optional) ────────────────────────────────────

uploaded_files = st.file_uploader(
    "Upload documents (optional)",
    type=["pdf", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="PDFs and spreadsheets will be added to the research database alongside web sources.",
)

# ── Input ─────────────────────────────────────────────────────

question = st.text_area(
    "Research question",
    height=90,
    placeholder="e.g. What are the most effective approaches to AI alignment research in 2026?",
)

run_clicked = st.button(
    "Research",
    type="primary",
    disabled=not question.strip(),
    use_container_width=False,
)

# ── Helper: render one agent step into the status widget ──────

def _render_step(status, event: dict) -> None:
    etype = event["type"]

    if etype == "tool_call":
        name = event["name"]
        args = event["args"]
        if name == "search_web":
            status.write(f"🔍 **Searching:** *{args['query']}*")
        elif name == "scrape_and_store":
            url = args["url"]
            short = url if len(url) <= 70 else url[:67] + "..."
            status.write(f"📄 **Reading:** {short}")
        elif name == "generate_report":
            status.write("✍️ **Generating report...**")

    elif etype == "tool_result":
        name   = event["name"]
        result = event["result"]
        if name == "scrape_and_store":
            status.write(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {result}")
        elif name == "search_web":
            n = result.count("URL:")
            status.write(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {n} results found")

    elif etype == "limit":
        status.write("⚠️ Agent reached the step limit.")


# ── Agent run ─────────────────────────────────────────────────

if run_clicked and question.strip():
    st.session_state.pop("report", None)
    st.session_state.pop("question", None)

    collection = create_collection()

    # Ingest uploaded files before the agent runs.
    if uploaded_files:
        with st.status("Reading uploaded files...", expanded=False) as ingest_status:
            for f in uploaded_files:
                name_lower = f.name.lower()
                if name_lower.endswith(".pdf"):
                    n = ingest_pdf(f.read(), f.name, collection)
                elif name_lower.endswith((".csv", ".xlsx", ".xls")):
                    n = ingest_csv(f.read(), f.name, collection)
                else:
                    n = 0
                ingest_status.write(f"📎 **{f.name}** → {n} chunks stored")
            ingest_status.update(label="Files ready.", state="complete")

    # Run the agent in a background thread so the Streamlit main thread
    # stays alive and keeps the WebSocket connection open.
    # The thread writes step events to a plain list (GIL-safe for appends);
    # the main thread reads from it and updates the status widget.
    steps: list[dict] = []
    outcome: dict = {}          # populated by worker: {"report": ...} or {"error": ...}

    def _worker():
        try:
            report = run_agent(
                question.strip(),
                collection,
                on_step=lambda e: steps.append(e),
            )
            outcome["report"] = report
        except Exception as exc:
            outcome["error"] = str(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    rendered = 0
    with st.status("Agent is researching...", expanded=True) as status:
        while thread.is_alive() or rendered < len(steps):
            # Flush any new step events to the status widget
            while rendered < len(steps):
                _render_step(status, steps[rendered])
                rendered += 1
            if thread.is_alive():
                time.sleep(0.3)   # yield so Streamlit can send WebSocket frames

        thread.join()

        if "error" in outcome:
            status.update(label="Error", state="error", expanded=True)
            st.error(f"Something went wrong: {outcome['error']}")
        else:
            st.session_state["report"]   = outcome["report"]
            st.session_state["question"] = question.strip()
            status.update(label="Research complete!", state="complete", expanded=False)

# ── Report display ────────────────────────────────────────────

if st.session_state.get("report"):
    st.divider()
    st.markdown(f"**Question:** {st.session_state['question']}")
    st.markdown("")
    st.markdown(st.session_state["report"])
    st.download_button(
        label="Download report (.md)",
        data=st.session_state["report"],
        file_name="research_report.md",
        mime="text/markdown",
    )
