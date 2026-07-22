import time

import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Graphitti", page_icon="🕸️", layout="wide")

if "current_source_url" not in st.session_state:
    st.session_state.current_source_url = None
if "last_ingest_stats" not in st.session_state:
    st.session_state.last_ingest_stats = None
if "graph_nonce" not in st.session_state:
    st.session_state.graph_nonce = str(time.time())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

STRATEGY_LABELS = {
    "dense_vector": "Dense (TF-IDF / embeddings)",
    "sparse_bm25": "Sparse BM25",
    "graph_traversal": "Graph traversal (multi-hop)",
    "graph_1hop": "Graph 1-hop",
    "entity_centric": "Entity-centric (full neighborhood)",
    "hybrid_fusion": "Hybrid fusion (RRF across all strategies)",
}


def backend_url() -> str:
    return st.session_state.get("backend_url", "http://localhost:8000").rstrip("/")


def call_ingest(url: str, depth: int, pages: int):
    resp = requests.post(
        f"{backend_url()}/ingest",
        json={"url": url, "depth": depth, "pages": pages, "reset": True},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def call_reset():
    resp = requests.post(f"{backend_url()}/reset", timeout=30)
    resp.raise_for_status()
    return resp.json()


def call_query(query: str):
    resp = requests.post(f"{backend_url()}/query", json={"query": query}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def fetch_graph():
    resp = requests.get(f"{backend_url()}/graph", timeout=30)
    resp.raise_for_status()
    return resp.json()


with st.sidebar:
    st.title("🕸️ Graphitti")
    st.caption("Graph-native web intelligence")

    st.text_input(
        "Backend URL",
        value=st.session_state.get("backend_url", "http://localhost:8000"),
        key="backend_url",
        help="Where graphitti.api.app (uvicorn) is running.",
    )

    st.divider()
    st.subheader("Ingest a page")
    ingest_url = st.text_input("URL to crawl", placeholder="https://en.wikipedia.org/wiki/...")
    col_depth, col_pages = st.columns(2)
    depth = col_depth.number_input("Depth", min_value=0, max_value=5, value=2)
    pages = col_pages.number_input("Max pages", min_value=1, max_value=100, value=20)

    st.caption(
        "Every ingest replaces the graph with the crawl of this URL only. "
        "The dashboard always shows exactly one entity's graph at a time."
    )

    if st.button("Crawl & Ingest", type="primary", use_container_width=True):
        if not ingest_url.strip():
            st.error("Enter a URL first.")
        else:
            with st.spinner(f"Crawling {ingest_url} and building its graph..."):
                try:
                    stats = call_ingest(ingest_url.strip(), int(depth), int(pages))
                    st.session_state.current_source_url = stats["source_url"]
                    st.session_state.last_ingest_stats = stats
                    st.session_state.graph_nonce = str(time.time())
                    st.session_state.chat_history = []
                    st.success(f"Graph ready for {stats['source_url']}")
                except requests.HTTPError as e:
                    st.error(f"Ingest failed: {e.response.text}")
                except requests.RequestException as e:
                    st.error(f"Could not reach backend: {e}")
            st.rerun()

    if st.button("Clear graph", use_container_width=True):
        try:
            call_reset()
            st.session_state.current_source_url = None
            st.session_state.last_ingest_stats = None
            st.session_state.graph_nonce = str(time.time())
            st.session_state.chat_history = []
            st.success("Graph cleared.")
        except requests.RequestException as e:
            st.error(f"Could not reach backend: {e}")
        st.rerun()

    if st.session_state.last_ingest_stats:
        s = st.session_state.last_ingest_stats
        st.divider()
        st.subheader("Last ingest")
        st.metric("Pages crawled", s["pages_crawled"])
        st.metric("Triples extracted", s["triples_extracted"])
        st.metric("Contested claim groups", s["contested_claim_groups"])
        st.caption(f"Spin-up time: {s['spin_up_seconds']}s")

tab_graph, tab_qa = st.tabs(["Knowledge Graph", "Ask a Question"])

with tab_graph:
    if st.session_state.current_source_url:
        st.subheader(f"Graph for: {st.session_state.current_source_url}")
    else:
        st.subheader("Knowledge Graph")

    refresh_col, _ = st.columns([1, 5])
    if refresh_col.button("Refresh graph"):
        st.session_state.graph_nonce = str(time.time())
        st.rerun()

    if not st.session_state.current_source_url:
        st.info("Ingest a URL from the sidebar to see its graph here.")
    else:
        try:
            graph_data = fetch_graph()
        except requests.RequestException as e:
            graph_data = None
            st.error(f"Could not reach backend: {e}")

        if graph_data is not None:
            n_nodes = len(graph_data.get("nodes", []))
            n_edges = len(graph_data.get("edges", []))
            m1, m2 = st.columns(2)
            m1.metric("Entities", n_nodes)
            m2.metric("Relations", n_edges)

            if n_nodes == 0:
                st.warning("No entities were extracted from this page.")
            else:
                view_url = f"{backend_url()}/graph/view?t={st.session_state.graph_nonce}"
                components.iframe(view_url, height=650, scrolling=False)

with tab_qa:
    st.subheader("Ask a question about the ingested graph")

    if not st.session_state.current_source_url:
        st.info("Ingest a URL first, then ask questions about it here.")
    else:
        st.caption(f"Answering from: {st.session_state.current_source_url}")

        question = st.text_input("Your question", key="qa_input", placeholder="e.g. Who is this person?")
        ask = st.button("Ask", type="primary")

        if ask and question.strip():
            with st.spinner("Routing, retrieving, and synthesizing an answer..."):
                try:
                    result = call_query(question.strip())
                    st.session_state.chat_history.insert(0, result)
                except requests.HTTPError as e:
                    st.error(f"Query failed: {e.response.text}")
                except requests.RequestException as e:
                    st.error(f"Could not reach backend: {e}")

        for result in st.session_state.chat_history:
            with st.container(border=True):
                st.markdown(f"**Q: {result['query']}**")
                st.markdown(result.get("answer") or "_No answer produced._")

                badges = []
                if result.get("intent"):
                    badges.append(f"intent: `{result['intent']}`")
                if result.get("strategy_used"):
                    label = STRATEGY_LABELS.get(result["strategy_used"], result["strategy_used"])
                    badges.append(f"strategy: `{label}`")
                if result.get("faithfulness") is not None:
                    badges.append(f"faithfulness: `{result['faithfulness']}`")
                if result.get("retries"):
                    badges.append(f"retries: `{result['retries']}`")
                if badges:
                    st.caption(" · ".join(badges))

                if result.get("contested_warnings"):
                    for w in result["contested_warnings"]:
                        st.warning(
                            f"Contested: ({w.get('subject')}, {w.get('predicate')}) "
                            f"-> {w.get('object')}"
                        )

                if result.get("evidence"):
                    with st.expander("Evidence used"):
                        for i, e in enumerate(result["evidence"], start=1):
                            st.markdown(f"**[{i}]** {e.get('text')}")
                            if e.get("source_url"):
                                st.caption(e["source_url"])

                if result.get("trace"):
                    with st.expander("Agent trace"):
                        for step in result["trace"]:
                            st.text(f"{step.get('agent')}: {step.get('note')}")
