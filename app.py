"""
app.py
Streamlit UI for the AI Research Assistant.

Flow:
  1. User types a research question and clicks "Research".
  2. A fresh ChromaDB collection is created for this session.
  3. The agent runs in a background thread so the main thread stays alive
     and keeps the WebSocket connection open with the browser.
  4. The main thread polls a shared event list and updates the status widget.
  5. When the agent finishes, the report is rendered directly in the same
     script pass — no st.rerun(). st.rerun() triggers a WebSocket reset
     that causes a "Connection error" flash and page reload, clearing session_state.
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


# ── State initialised before the conditional blocks ───────────
# _fresh_report is populated if the agent just ran in THIS script pass.
# The report-display section reads it first, falling back to session_state
# for page reloads. This avoids needing st.rerun() which resets the WebSocket.

_fresh_report    = ""
_fresh_question  = ""

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
    steps: list[dict] = []
    outcome: dict = {}

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
    deadline = time.time() + 360

    with st.status("Agent is researching...", expanded=True) as status:
        while thread.is_alive() or rendered < len(steps):
            while rendered < len(steps):
                _render_step(status, steps[rendered])
                rendered += 1
            if thread.is_alive():
                if time.time() > deadline:
                    outcome["error"] = (
                        "Timed out after 6 minutes. "
                        "Try a more specific question or fewer sources."
                    )
                    break
                time.sleep(0.3)

        thread.join(timeout=5)

        if "error" in outcome:
            status.update(label="Error", state="error", expanded=True)
            st.error(f"Something went wrong: {outcome['error']}")
        else:
            st.session_state["report"]   = outcome["report"]
            st.session_state["question"] = question.strip()
            status.update(label="Research complete!", state="complete", expanded=False)
            _fresh_report   = outcome["report"]
            _fresh_question = question.strip()

# ── Report display ────────────────────────────────────────────

_report_to_show   = _fresh_report   or st.session_state.get("report",   "")
_question_to_show = _fresh_question or st.session_state.get("question", "")

if _report_to_show:
    st.divider()
    st.markdown(f"**Question:** {_question_to_show}")
    st.markdown("")
    st.markdown(_report_to_show)
    st.download_button(
        label="Download report (.md)",
        data=_report_to_show,
        file_name="research_report.md",
        mime="text/markdown",
    )
