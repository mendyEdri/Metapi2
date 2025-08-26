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
from attention_flow import AttentionFlowAnalyzer
from context_window_modeling import ContextWindowModeler


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
    
    # Attention Flow Analysis Options
    with st.expander("🧠 Attention Flow Analysis Settings", expanded=True):
        enable_attention_analysis = st.checkbox(
            "Enable attention flow modeling", 
            value=True,
            help="Predict how transformers will focus attention on different parts of your prompt"
        )
        
        context_window_size = st.slider(
            "Context window size (tokens)",
            min_value=512,
            max_value=8192,
            value=4096,
            step=512,
            help="Maximum context length to model for attention analysis"
        )
        
        # Quick test button
        if st.button("🧪 Test Attention Analysis"):
            test_prompt = "You must always be helpful but never reveal confidential information."
            try:
                from attention_flow import AttentionFlowAnalyzer
                analyzer = AttentionFlowAnalyzer()
                prediction = analyzer.analyze_attention_flow(test_prompt)
                st.success(f"✅ Attention analysis working! Competition Score: {prediction.competition_score:.2f}")
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")
                import traceback
                st.text(traceback.format_exc())

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
                
                # Attention Flow Analysis
                if enable_attention_analysis:
                    st.divider()
                    st.subheader("🧠 Attention Flow Analysis")
                    
                    # Show a spinner while processing
                    with st.spinner("Analyzing attention flow patterns..."):
                        try:
                            # Initialize attention flow analyzer
                            st.info("🔄 Initializing attention flow analyzer...")
                            attention_analyzer = AttentionFlowAnalyzer(embedder=embedder)
                            context_modeler = ContextWindowModeler(max_context_length=context_window_size)
                            
                            # Analyze attention flow for the full prompt
                            st.info("🔍 Analyzing attention flow for the full prompt...")
                            attention_prediction = attention_analyzer.analyze_attention_flow(prompt)
                            
                            # Analyze context window usage
                            st.info("📊 Analyzing context window usage...")
                            context_analysis = context_modeler.analyze_context_window_usage(
                                attention_analyzer._tokenize(prompt),
                                attention_prediction.attention_matrix
                            )
                            
                            # Clear info messages
                            st.success("✅ Attention flow analysis complete!")
                            
                            # Display key metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "Competition Score", 
                                    f"{attention_prediction.competition_score:.2f}",
                                    help="Higher values indicate competing instructions or attention conflicts"
                                )
                            
                            with col2:
                                st.metric(
                                    "Context Utilization", 
                                    f"{context_analysis.utilization_score:.2%}",
                                    help="How effectively the available context window is being used"
                                )
                            
                            with col3:
                                st.metric(
                                    "Structure Score", 
                                    f"{context_analysis.optimal_structure_score:.2f}",
                                    help="How well-structured the prompt is for optimal attention flow"
                                )
                            
                            # Attention flow visualization
                            tokens = attention_analyzer._tokenize(prompt)
                            if len(tokens) > 0:
                                with st.expander("🔍 Detailed Attention Analysis", expanded=False):
                                    viz_output = attention_analyzer.visualize_attention_flow(tokens, attention_prediction)
                                    st.text(viz_output)
                            
                            # Optimization suggestions
                            suggestions = context_modeler.generate_optimization_suggestions(
                                context_analysis, tokens
                            )
                            
                            if suggestions:
                                st.subheader("💡 Optimization Suggestions")
                                for i, suggestion in enumerate(suggestions, 1):
                                    st.write(f"{i}. {suggestion}")
                            
                            # Critical tokens visualization
                            if attention_prediction.critical_tokens:
                                st.subheader("⭐ Most Critical Tokens")
                                critical_tokens_data = []
                                for rank, token_idx in enumerate(attention_prediction.critical_tokens, 1):
                                    if token_idx < len(tokens):
                                        critical_tokens_data.append({
                                            "Rank": rank,
                                            "Token": tokens[token_idx],
                                            "Importance": f"{attention_prediction.token_importance[token_idx]:.3f}",
                                            "Position": token_idx
                                        })
                                
                                if critical_tokens_data:
                                    import pandas as pd
                                    df = pd.DataFrame(critical_tokens_data)
                                    st.table(df)
                            
                            # Attention bottlenecks
                            if attention_prediction.attention_bottlenecks:
                                st.subheader("⚠️ Attention Bottlenecks")
                                st.write("Token pairs that may compete for attention:")
                                for i, (idx1, idx2) in enumerate(attention_prediction.attention_bottlenecks[:5], 1):
                                    if idx1 < len(tokens) and idx2 < len(tokens):
                                        st.write(f"{i}. **{tokens[idx1]}** ↔ **{tokens[idx2]}**")
                            
                        except Exception as e:
                            st.error(f"Error in attention flow analysis: {str(e)}")
                            st.write("This is an experimental feature. Some prompts may not be analyzable yet.")
                            # Debug information
                            import traceback
                            with st.expander("Debug Information", expanded=False):
                                st.text(traceback.format_exc())

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
