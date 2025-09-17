# app/streamlit_app.py
"""
Streamlit front-end for semantic search demo using a vector database (FAISS + SQLite).

Features included:
- Query expansion (WordNet synonyms)
- Fuzzy matching + "Did you mean?" suggestions
- Semantic search with SentenceTransformers embeddings
- Optional re-ranking with a cross-encoder
- Clustering of retrieved results
- Extractive summarization of results
- Top-K explanation plot
- Expandable result explorer with metadata, fuzzy scores, raw scores, and neighbors
"""

import streamlit as st
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure project root is available on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Light-weight imports only here (heavy ML libs are lazy-imported later)
from src.utils import ensure_metadata_table

# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Vector DB Demo — Advanced")
st.title("Vector DB Demo — Advanced Features")

# Ensure the SQLite metadata table exists (id, title, text, etc.)
ensure_metadata_table()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
col_main, col_side = st.columns([3, 1])

with col_side:
    st.subheader("Search options")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
    use_expansion = st.checkbox("Query expansion (WordNet)", value=True)
    rerank = st.checkbox("Re-rank with cross-encoder", value=True)
    min_score = st.slider("Min re-rank score threshold", 0.0, 1.0, 0.0, 0.01)
    show_all = st.checkbox("Show raw scores table", value=False)

    st.markdown("---")

    st.subheader("Tools")
    if st.button("Reload modules"):
        # Streamlit-specific: reruns the script (helpful after fixing env vars)
        st.experimental_rerun()

    st.caption("Use 'Reload modules' after updating code or fixing env vars. See README for troubleshooting.")

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
with col_main:
    query = st.text_input("Enter search (try: airport, traffic, luggage)", "")
    search_btn = st.button("Search")

# ---------------------------------------------------------------------------
# Trigger search pipeline when user clicks "Search"
# ---------------------------------------------------------------------------
if search_btn and query.strip():
    # Lazy import: avoid loading ML libs at Streamlit startup
    from src.query import search

    # Run the semantic search pipeline
    res = search(
        query,
        k=top_k,
        min_score=min_score,
        use_expansion=use_expansion,
        rerank=rerank,
        cluster=True,  # enable clustering of results
    )

    results = res.get("results", [])
    suggestions = res.get("suggestions", [])
    summary = res.get("summary", "")
    explanations = res.get("explanations", [])
    expanded_terms = res.get("expanded_terms", [])

    # -----------------------------------------------------------------------
    # Show "Did you mean?" fuzzy suggestions
    # -----------------------------------------------------------------------
    if suggestions:
        st.info("Did you mean: " + ", ".join([f"{s[0]} ({s[1]}%)" for s in suggestions]))

    # Show which expansion terms were added to the query
    if expanded_terms:
        st.caption("Query expansions added: " + ", ".join(expanded_terms))

    # -----------------------------------------------------------------------
    # Summarization of retrieved results
    # -----------------------------------------------------------------------
    if summary:
        st.subheader("Summary of top results")
        st.write(summary)

    # -----------------------------------------------------------------------
    # Top-K explanation (plot score curve)
    # -----------------------------------------------------------------------
    if explanations:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(explanations) + 1), explanations, marker="o")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Score")
        ax.set_title("Top-K score curve")
        st.pyplot(fig)

    # -----------------------------------------------------------------------
    # Result Explorer: expandable cards per result
    # -----------------------------------------------------------------------
    st.subheader("Search results")
    for i, r in enumerate(results):
        title = r.get("title") or f"Document {r.get('id')}"
        score = r.get("rerank_score", r.get("score", 0.0))
        cluster = r.get("cluster", None)

        expander_label = (
            f"{i+1}. {title}  — score: {score:.3f}"
            + (f"  — cluster: {cluster}" if cluster is not None else "")
        )

        with st.expander(expander_label, expanded=(i < 2)):
            # Show highlighted text (semantic matches highlighted)
            st.markdown(r.get("highlighted_text", r.get("text", "")))
            st.write("")  # spacer

            # Show metadata in columns
            cols = st.columns([1, 2, 2])
            with cols[0]:
                st.write("ID")
                st.code(r.get("id"))
            with cols[1]:
                st.write("Fuzzy score")
                st.write(f"{r.get('fuzzy_score', 0.0):.3f}")
            with cols[2]:
                st.write("Raw score")
                st.write(f"{r.get('score', 0.0):.3f}")

            # Action buttons for exploration / annotation
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button(f"Show neighbors_{r['id']}", key=f"nb_{i}"):
                    # Show nearest neighbors of this document in vector space
                    from src.embed import embed_texts
                    from src.indexer import get_index_and_doc_ids

                    idx, doc_ids = get_index_and_doc_ids()
                    try:
                        pos = doc_ids.index(r["id"])
                        vec = idx.reconstruct(pos).reshape(1, -1)
                        D, I = idx.search(vec, 6)
                        neigh_ids = [
                            doc_ids[j] for j in I[0] if j != pos and j != -1
                        ][:5]
                        st.write("Neighbors:", neigh_ids)
                    except Exception as e:
                        st.error(f"Unable to fetch neighbors: {e}")
            with c2:
                if st.button(f"Add note to {r['id']}", key=f"note_{i}"):
                    st.info("Feature placeholder: integrate annotation store (not yet implemented).")

    # -----------------------------------------------------------------------
    # Show raw result dataframe (optional)
    # -----------------------------------------------------------------------
    if show_all:
        import pandas as pd

        if results:
            df = pd.DataFrame(results)
            st.subheader("Raw results table")
            st.dataframe(df)
        else:
            st.info("No results to display.")

# ---------------------------------------------------------------------------
# Initial state (no query entered)
# ---------------------------------------------------------------------------
else:
    if not query.strip():
        st.info("Enter a query and click Search to start semantic retrieval.")
