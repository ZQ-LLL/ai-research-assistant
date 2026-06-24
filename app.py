"""
app.py
Streamlit UI for the AI Research Assistant.

Flow:
  1. User types a research question and clicks "Research".
  2. A fresh ChromaDB collection is created for this session.
  3. The agent loop runs — each tool call is shown live in a status widget.
  4. When the agent calls generate_report, the report is stored in session state.
  5. The report is rendered as markdown with a download button.
"""

import streamlit as st

from utils.agent import run_agent
from utils.ingest import ingest_pdf
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
    type=["pdf"],
    accept_multiple_files=True,
    help="PDFs will be added to the research database alongside web sources.",
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

# ── Agent run ─────────────────────────────────────────────────

if run_clicked and question.strip():
    # Clear any previous report
    st.session_state.pop("report", None)
    st.session_state.pop("question", None)

    collection = create_collection()

    # Ingest uploaded files before the agent runs so it can retrieve
    # content from them via query_chunks, same as any web source.
    if uploaded_files:
        with st.status("Reading uploaded files...", expanded=False) as ingest_status:
            for f in uploaded_files:
                n = ingest_pdf(f.read(), f.name, collection)
                ingest_status.write(f"📎 **{f.name}** → {n} chunks stored")
            ingest_status.update(label="Files ready.", state="complete")

    with st.status("Agent is researching...", expanded=True) as status:

        def on_step(event: dict):
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
                name  = event["name"]
                result = event["result"]

                if name == "scrape_and_store":
                    # Show chunk count (or failure) indented under the URL
                    status.write(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {result}")

                elif name == "search_web":
                    n = result.count("URL:")
                    status.write(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {n} results found")

            elif etype == "limit":
                st.warning("Agent reached the step limit without finishing.")

        try:
            report = run_agent(question.strip(), collection, on_step=on_step)
            st.session_state["report"]   = report
            st.session_state["question"] = question.strip()
            status.update(label="Research complete!", state="complete", expanded=False)

        except Exception as e:
            status.update(label="Error", state="error", expanded=True)
            st.error(f"Something went wrong: {e}")

# ── Report display ────────────────────────────────────────────

if st.session_state.get("report"):
    st.divider()

    # Question recap
    st.markdown(f"**Question:** {st.session_state['question']}")
    st.markdown("")

    # Report body
    st.markdown(st.session_state["report"])

    # Download button
    st.download_button(
        label="Download report (.md)",
        data=st.session_state["report"],
        file_name="research_report.md",
        mime="text/markdown",
    )
