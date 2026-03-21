"""
streamlit_app.py
----------------
Frontend for the Legal Precedent RAG API.
Calls the FastAPI backend running locally or on AWS.

Run locally (API must be running on port 8000):
    streamlit run streamlit_app.py

For AWS deployment, set API_BASE_URL to your EC2 public IP:
    API_BASE_URL=http://YOUR_EC2_IP:8000 streamlit run streamlit_app.py
"""

import os
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Legal Precedent Research",
    page_icon  = "⚖️",
    layout     = "wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .case-card {
        background: #f8f9fa;
        border-left: 4px solid #1f618d;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .case-name {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1f618d;
        margin-bottom: 0.3rem;
    }
    .case-meta {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    .score-badge {
        display: inline-block;
        background: #1f618d;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .memo-box {
        background: #fdfefe;
        border: 1px solid #d5e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        line-height: 1.8;
        font-size: 0.95rem;
    }
    .metric-box {
        background: #eaf2ff;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("## ⚖️ Legal Precedent Research")
st.markdown(
    "Enter a legal query to retrieve relevant US court precedents "
    "and generate a structured legal research memo."
)
st.markdown("---")

# ── Health check sidebar ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### System Status")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        st.success("API connected")
        st.markdown(f"**Qdrant:** {health.get('qdrant', 'unknown')}")
        st.markdown(f"**Vectors:** {health.get('vectors_count', 0):,}")
        st.markdown(f"**Model:** {health.get('ollama_model', 'unknown')}")
    except Exception:
        st.error("API not reachable")
        st.markdown(f"Expected at: `{API_BASE_URL}`")
        st.markdown("Start the API: `uvicorn api:app --port 8000`")

    st.markdown("---")
    st.markdown("### Settings")
    top_k    = st.slider("Cases to retrieve", 1, 10, 5)
    generate = st.toggle("Generate legal memo", value=True)

    st.markdown("---")
    st.markdown(
        "Built on [arXiv:2406.01609](https://arxiv.org/abs/2406.01609)  \n"
        "Stack: LangChain · Qdrant · FastAPI · Ollama"
    )

# ── Query input ───────────────────────────────────────────────────────────────

example_queries = [
    "Select an example query...",
    "Fourth Amendment unlawful search and seizure warrant requirement",
    "First Amendment freedom of speech government restriction",
    "Equal protection racial discrimination employment civil rights",
    "Due process right to fair trial criminal defendant",
    "Habeas corpus unlawful detention prisoner rights",
]

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_area(
        "Legal query",
        placeholder="e.g. Fourth Amendment search and seizure warrant requirement...",
        height=100,
        label_visibility="collapsed",
    )
with col2:
    example = st.selectbox("Or try an example", example_queries,
                           label_visibility="collapsed")
    if example != example_queries[0]:
        query = example

search_clicked = st.button("🔍 Search Precedents", type="primary",
                            disabled=not query or len(query.strip()) < 10)

if query and len(query.strip()) < 10:
    st.warning("Query must be at least 10 characters.")

# ── Results ───────────────────────────────────────────────────────────────────

if search_clicked and query and len(query.strip()) >= 10:

    with st.spinner("Retrieving precedents..."):
        try:
            endpoint = "/query" if generate else "/retrieve"
            response = requests.post(
                f"{API_BASE_URL}{endpoint}",
                json={"query": query.strip(), "top_k": top_k, "generate": generate},
                timeout=180,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_BASE_URL}. Is it running?")
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timed out. The LLM may be slow — try with memo generation off.")
            st.stop()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.stop()

    # Timing metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Cases found", len(data.get("cases", [])))
    with m2:
        st.metric("Retrieval", f"{data.get('retrieval_ms', 0)}ms")
    with m3:
        if data.get("generation_ms"):
            st.metric("Generation", f"{data.get('generation_ms', 0)}ms")
        else:
            st.metric("Generation", "skipped")
    with m4:
        st.metric("Total", f"{data.get('total_ms', 0)}ms")

    st.markdown("---")

    # Two column layout — cases left, memo right
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Retrieved Cases")
        cases = data.get("cases", [])
        if not cases:
            st.info("No relevant cases found. Try rephrasing your query.")
        else:
            for i, case in enumerate(cases, 1):
                score_pct = int(case["score"] * 100)
                st.markdown(f"""
                <div class="case-card">
                    <div class="case-name">[{i}] {case['case_name']}</div>
                    <div class="case-meta">
                        {case.get('court', 'Unknown court')} &nbsp;·&nbsp;
                        {case.get('date_filed', '')[:10]} &nbsp;·&nbsp;
                        {case.get('author_name', '')}
                    </div>
                    <span class="score-badge">Relevance {score_pct}%</span>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"View excerpt — {case['case_name'][:40]}..."):
                    st.markdown(case.get("chunk_preview", "No preview available"))
                    if case.get("absolute_url"):
                        st.markdown(f"[View full opinion ↗]({case['absolute_url']})")

    with right:
        st.markdown("### Legal Research Memo")
        memo = data.get("memo")
        if not generate:
            st.info("Memo generation is off. Toggle it on in the sidebar.")
        elif not memo:
            st.warning("No memo was generated.")
        elif memo.startswith("ERROR"):
            st.error(f"LLM error: {memo}")
            st.info("Try turning off memo generation — retrieval still works.")
        else:
            st.markdown(
                f'<div class="memo-box">{memo.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                label     = "📄 Download memo",
                data      = memo,
                file_name = "legal_research_memo.txt",
                mime      = "text/plain",
            )

# ── Empty state ───────────────────────────────────────────────────────────────

else:
    if not search_clicked:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; color: #888;">
            <div style="font-size:3rem">⚖️</div>
            <div style="font-size:1.1rem; margin-top:1rem">
                Enter a legal query above to get started
            </div>
            <div style="font-size:0.9rem; margin-top:0.5rem">
                Searches 22,809 US court opinions across federal and state courts
            </div>
        </div>
        """, unsafe_allow_html=True)