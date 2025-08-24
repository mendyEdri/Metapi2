"""Streamlit app demonstrating OpenAI embeddings via LangChain.

The app provides a settings modal where users can supply an OpenAI API key.
The key is persisted in the browser's local storage so it only needs to be
entered once. When a key is available an example embedding for "Hello world" is
generated and displayed.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from streamlit_js_eval import streamlit_js_eval
from clustering import (
    cluster_embeddings,
    visualize_clusters,
    build_chunk_graph,
    compute_chunk_weights,
)
from prompt_chunking import chunk_prompt


st.title("Hello, World!")


def is_valid_openai_api_key(key: str) -> bool:
    """Basic OpenAI API key validation."""
    return isinstance(key, str) and key.startswith("sk-") and len(key) > 40


def load_api_key() -> str:
    """Retrieve the stored API key from local storage."""
    api_key = streamlit_js_eval(
        js_expressions="localStorage.getItem('openai_api_key')",
        key="get_api_key",
    )
    return api_key or ""


def save_api_key(key: str) -> None:
    """Persist the API key to local storage."""
    streamlit_js_eval(
        js_expressions=f"localStorage.setItem('openai_api_key', {json.dumps(key)})",
        key="set_api_key",
    )


if "openai_api_key" not in st.session_state:
    # Initialize the key in session state; the actual value will be
    # loaded from the browser's local storage on each rerun.
    st.session_state.openai_api_key = ""

# `streamlit_js_eval` resolves asynchronously, so we fetch the value on
# every run and update session state when a stored key becomes available.
stored_api_key = load_api_key()
if stored_api_key and stored_api_key != st.session_state.openai_api_key:
    st.session_state.openai_api_key = stored_api_key


def open_settings() -> None:
    st.session_state.show_settings = True


def close_settings() -> None:
    st.session_state.show_settings = False


st.button("Settings", on_click=open_settings)

if st.session_state.get("show_settings"):
    has_modal = hasattr(st, "modal")
    container = st.modal("Settings") if has_modal else st.sidebar
    with container:
        if not has_modal:
            st.header("Settings")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
        )

        col_save, col_close = st.columns(2)
        with col_save:
            if st.button("Save"):
                if is_valid_openai_api_key(api_key_input):
                    st.session_state.openai_api_key = api_key_input
                    save_api_key(api_key_input)
                    close_settings()
                else:
                    st.error(
                        "Invalid API key format. Please enter a valid OpenAI API key."
                    )
        with col_close:
            st.button("Close", on_click=close_settings)


api_key = st.session_state.openai_api_key

if api_key:
    model_name = "text-embedding-3-small"
    # Cache the embedder in session_state to avoid recreating it unnecessarily
    if (
        "embedder" not in st.session_state
        or st.session_state.get("embedder_api_key") != api_key
        or st.session_state.get("embedder_model") != model_name
    ):
        st.session_state.embedder = OpenAIEmbeddings(
            model=model_name, openai_api_key=api_key
        )
        st.session_state.embedder_api_key = api_key
        st.session_state.embedder_model = model_name
    embedder = st.session_state.embedder

    prompt = st.text_area("System Prompt")

    algorithm = st.selectbox(
        "Clustering algorithm", ["kmeans", "agglomerative", "dbscan"], index=0
    )
    n_clusters = None
    if algorithm != "dbscan":
        n_clusters = st.number_input(
            "Number of clusters", min_value=2, max_value=10, value=3
        )

    weight_method = st.selectbox(
        "Chunk weight method",
        ["cosine", "center"],
        index=0,
        help="`cosine` weights by similarity to the centroid; `center` downweights outliers",
    )

    def cluster_prompt(text: str):
        chunks = chunk_prompt(text)
        if len(chunks) < 2:
            raise ValueError("Need at least two chunks for clustering.")
        vectors = [embedder.embed_query(chunk) for chunk in chunks]
        embeddings = np.array(vectors)
        k = int(n_clusters or 3)
        if algorithm != "dbscan":
            k = min(k, len(embeddings))
        labels = cluster_embeddings(embeddings, algorithm=algorithm, n_clusters=k)
        weights = compute_chunk_weights(embeddings, method=weight_method)
        return chunks, embeddings, labels, weights

    if st.button("Analyze prompt"):
        if prompt.strip():
            try:
                chunks, embeddings, labels, weights = cluster_prompt(prompt)
            except ValueError as err:
                st.error(str(err))
            else:
                st.subheader("Chunk details")
                for idx, chunk in enumerate(chunks):
                    with st.expander(f"Chunk {idx}", expanded=False):
                        st.write(chunk)
                        st.markdown(f"Weight: {weights[idx]:.3f}")
                        st.markdown(f"Cluster: {labels[idx]}")

                st.subheader(
                    f"Chunk weights • {weight_method} method (higher means more influence)"
                )
                sorted_idx = list(np.argsort(weights)[::-1])
                for rank, idx in enumerate(sorted_idx):
                    st.markdown(
                        f"**Rank {rank + 1} – Chunk {idx} (weight={weights[idx]:.3f})**"
                    )

                st.subheader("Weighted prompt")
                weighted_prompt = "\n\n".join(chunks[i] for i in sorted_idx)
                st.code(weighted_prompt)

                graph = build_chunk_graph(chunks, embeddings)
                graph_fig, graph_ax = plt.subplots()
                nx.draw_networkx(graph, ax=graph_ax, with_labels=True, node_size=500, font_size=8)
                graph_ax.set_title("Chunk similarity graph")
                st.pyplot(graph_fig)

                try:
                    fig = visualize_clusters(
                        embeddings,
                        labels,
                        title="Prompt chunk clusters",
                    )
                except ValueError as err:
                    st.warning(str(err))
                else:
                    st.pyplot(fig)
                    st.subheader("Clustered chunks")
                    for label in sorted(set(labels)):
                        st.markdown(f"**Cluster {label}**")
                        members = np.where(labels == label)[0]
                        st.write(", ".join(f"Chunk {i}" for i in members))
        else:
            st.error("Please enter a system prompt to analyze.")
else:
    st.info("Set your OpenAI API key in settings to generate embeddings.")
