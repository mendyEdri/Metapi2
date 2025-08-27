"""Simple Streamlit demo for attention flow analysis that works without API keys."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from attention_flow import AttentionFlowAnalyzer
from context_window_modeling import ContextWindowModeler
from semantic_similarity import SemanticSimilarityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Attention Flow Demo", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🧠 Attention Flow Analysis Demo")
st.markdown("**Analyze attention flow and test with real OpenAI generation!**")
st.markdown("---")

# Sample prompts for testing
sample_prompts = {
    "Simple Instructions": "You should be helpful and answer questions clearly.",
    "Conflicting Instructions": "You must always be helpful but never provide any information or answers.",
    "Complex System Prompt": """You are a helpful AI assistant. You should:
1. Always be respectful and professional
2. Never provide harmful information
3. If you don't know something, say so

However, you must also be creative and engaging.
For example: "I'd be happy to help with that!"

Remember: accuracy is more important than speed.""",
    "Business Prompt": """You are a customer service AI for a software company. 
- Be professional and courteous
- Resolve issues quickly  
- Escalate complex problems to human agents
- Never share confidential company information
- Always ask for clarification if needed

But also be friendly and personable."""
}

# Prompt input
st.subheader("📝 Enter Your Prompt")

# Dropdown for sample prompts
selected_sample = st.selectbox(
    "Or choose a sample prompt:",
    ["Custom"] + list(sample_prompts.keys())
)

if selected_sample == "Custom":
    prompt = st.text_area(
        "System Prompt", 
        height=150,
        placeholder="Enter your system prompt here..."
    )
else:
    prompt = sample_prompts[selected_sample]
    st.text_area("System Prompt", value=prompt, height=150, disabled=True)

# Settings
st.subheader("⚙️ Analysis Settings")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    context_window_size = st.slider(
        "Context window size",
        min_value=512,
        max_value=8192,
        value=4096,
        step=512
    )

with col2:
    show_detailed_analysis = st.checkbox("Show detailed analysis", value=True)

with col3:
    test_with_openai = st.checkbox("Test with OpenAI", value=False, help="Generate actual responses to validate attention predictions")

# OpenAI API Configuration
if test_with_openai:
    with st.expander("🔑 OpenAI API Configuration", expanded=True):
        col_api1, col_api2 = st.columns([2, 1])
        
        with col_api1:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key to test real generations"
            )
        
        with col_api2:
            model_name = st.selectbox(
                "Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=0,
                help="Choose the OpenAI model for testing"
            )
        
        test_questions = st.text_area(
            "Test Questions (one per line)",
            value="What is your primary language?\nCan you respond in Spanish?\nHow should you communicate?",
            height=80,
            help="Questions to test how the prompt affects responses"
        )

st.markdown("---")

# Analysis button
st.markdown("### 🚀 Analysis")
if st.button("🚀 Analyze Attention Flow", type="primary", use_container_width=True):
    if not prompt.strip():
        st.error("Please enter a prompt to analyze!")
    else:
        try:
            # Initialize analyzers
            with st.spinner("Analyzing attention flow..."):
                analyzer = AttentionFlowAnalyzer()  # No embedder needed for basic analysis
                context_modeler = ContextWindowModeler(max_context_length=context_window_size)
                
                # Analyze attention flow at chunk level
                prediction = analyzer.analyze_attention_flow(prompt, use_chunks=True)
                
                # Get chunks for context analysis
                from prompt_chunking import chunk_prompt
                chunks = chunk_prompt(prompt)
                if len(chunks) == 0:
                    chunks = [prompt]  # Fallback to full text as single chunk
                
                # Store embedder in session state for heatmap access
                if test_with_openai and api_key:
                    try:
                        from langchain_openai import OpenAIEmbeddings
                        embedder = OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-3-small")
                        st.session_state['embedder'] = embedder
                    except Exception:
                        st.session_state['embedder'] = None
                else:
                    st.session_state['embedder'] = None
                    
                context_analysis = context_modeler.analyze_context_window_usage(
                    chunks, prediction.attention_matrix
                )
            
            # Display results
            st.success("✅ Analysis Complete!")
            st.markdown("---")
            
            # Key metrics
            st.subheader("📊 Key Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Competition Score", 
                    f"{prediction.competition_score:.2f}",
                    help="Higher values indicate competing instructions or attention conflicts"
                )
            
            with col2:
                st.metric(
                    "Context Utilization", 
                    f"{context_analysis.utilization_score:.1%}",
                    help="How effectively the available context window is being used"
                )
            
            with col3:
                st.metric(
                    "Structure Score", 
                    f"{context_analysis.optimal_structure_score:.2f}",
                    help="How well-structured the prompt is for optimal attention flow"
                )
            
            st.markdown("---")
            
            # Critical chunks
            if prediction.critical_tokens:
                st.subheader("⭐ Most Critical Chunks")
                critical_data = []
                for rank, chunk_idx in enumerate(prediction.critical_tokens, 1):
                    if chunk_idx < len(chunks):
                        chunk_preview = chunks[chunk_idx][:100] + "..." if len(chunks[chunk_idx]) > 100 else chunks[chunk_idx]
                        chunk_preview = chunk_preview.replace('\n', ' ').strip()
                        critical_data.append({
                            "Rank": rank,
                            "Chunk Preview": chunk_preview,
                            "Importance": f"{prediction.token_importance[chunk_idx]:.3f}",
                            "Position": chunk_idx + 1
                        })
                
                if critical_data:
                    import pandas as pd
                    df = pd.DataFrame(critical_data)
                    st.dataframe(df, width="stretch")
            
            st.markdown("---")
            
            # Optimization suggestions
            suggestions = context_modeler.generate_optimization_suggestions(
                context_analysis, chunks
            )
            
            if suggestions:
                st.subheader("💡 Optimization Suggestions")
                for i, suggestion in enumerate(suggestions, 1):
                    st.info(f"**{i}.** {suggestion}")
            
            st.markdown("---")
            
            # Attention bottlenecks
            if prediction.attention_bottlenecks:
                st.subheader("⚠️ Attention Bottlenecks")
                st.write("Chunk pairs that may compete for attention:")
                bottleneck_data = []
                for i, (idx1, idx2) in enumerate(prediction.attention_bottlenecks[:5], 1):
                    if idx1 < len(chunks) and idx2 < len(chunks):
                        chunk1_preview = chunks[idx1][:50] + "..." if len(chunks[idx1]) > 50 else chunks[idx1]
                        chunk2_preview = chunks[idx2][:50] + "..." if len(chunks[idx2]) > 50 else chunks[idx2]
                        chunk1_preview = chunk1_preview.replace('\n', ' ').strip()
                        chunk2_preview = chunk2_preview.replace('\n', ' ').strip()
                        bottleneck_data.append({
                            "Conflict #": i,
                            "Chunk 1": chunk1_preview,
                            "Chunk 2": chunk2_preview,
                            "Position 1": idx1 + 1,
                            "Position 2": idx2 + 1
                        })
                
                if bottleneck_data:
                    import pandas as pd
                    df = pd.DataFrame(bottleneck_data)
                    st.dataframe(df, width="stretch")
            
            st.markdown("---")
            
            # Detailed analysis
            if show_detailed_analysis:
                st.subheader("🔍 Detailed Analysis")
                
                tab1, tab2 = st.tabs(["📊 Attention Flow", "📈 Chunk Statistics"])
                
                with tab1:
                    viz_output = analyzer.visualize_attention_flow(chunks, prediction, use_chunks=True)
                    st.text(viz_output)
                    
                with tab2:
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric("Total Chunks", len(chunks))
                    
                    with col_stat2:
                        st.metric("Effective Length", context_analysis.effective_length)
                    
                    with col_stat3:
                        st.metric("Bottleneck Positions", len(context_analysis.bottleneck_positions))
                    
                    # Show all chunks with their content
                    st.write("**Chunk Breakdown:**")
                    for i, chunk in enumerate(chunks):
                        importance = prediction.token_importance[i] if i < len(prediction.token_importance) else 0
                        with st.expander(f"Chunk {i+1} (Importance: {importance:.3f})", expanded=False):
                            st.code(chunk, language="text")
                    
                    # Chunk importance distribution
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    positions = np.arange(len(prediction.token_importance))
                    bars = ax.bar(positions, prediction.token_importance)
                    ax.set_xlabel("Chunk Position")
                    ax.set_ylabel("Importance Score")
                    ax.set_title("Chunk Importance Distribution")
                    
                    # Highlight critical chunks
                    for idx in prediction.critical_tokens[:3]:
                        if idx < len(positions):
                            bars[idx].set_color('red')
                            bars[idx].set_alpha(0.7)
                    
                    # Add chunk labels
                    for i, (bar, chunk) in enumerate(zip(bars, chunks)):
                        if i < 10:  # Only label first 10 chunks to avoid crowding
                            chunk_label = chunk[:20] + "..." if len(chunk) > 20 else chunk
                            chunk_label = chunk_label.replace('\n', ' ').strip()
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                                   chunk_label, rotation=45, ha='left', va='bottom', fontsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add chunk embeddings heatmap with semantic analysis
                    st.subheader("🔥 Chunk Similarity Analysis")
                    
                    # Add similarity method selection
                    similarity_col1, similarity_col2 = st.columns([2, 1])
                    with similarity_col1:
                        similarity_method = st.selectbox(
                            "Similarity Method",
                            ["Semantic (Meaning-based)", "Syntactic (Structure-based)", "Compare Both"],
                            index=0,
                            help="Choose how to measure chunk similarity"
                        )
                    
                    with similarity_col2:
                        if similarity_method == "Compare Both":
                            show_comparison = True
                        else:
                            show_comparison = st.checkbox("Show method comparison", value=False)
                    
                    if st.session_state.get('embedder') is not None:
                        try:
                            embedder = st.session_state.embedder
                            semantic_analyzer = SemanticSimilarityAnalyzer()
                            
                            with st.spinner("Computing semantic similarity..."):
                                if similarity_method == "Compare Both" or show_comparison:
                                    # Compute all methods for comparison
                                    comparison_results = semantic_analyzer.compare_similarity_methods(chunks, embedder)
                                    
                                    if similarity_method == "Semantic (Meaning-based)":
                                        similarity_matrix, processed_chunks = comparison_results['concepts']
                                        method_used = "concepts"
                                    elif similarity_method == "Syntactic (Structure-based)":
                                        similarity_matrix, processed_chunks = comparison_results['syntactic']
                                        method_used = "syntactic"
                                    else:  # Compare Both
                                        similarity_matrix, processed_chunks = comparison_results['concepts']
                                        method_used = "concepts"
                                else:
                                    # Single method computation
                                    if similarity_method == "Semantic (Meaning-based)":
                                        similarity_matrix, processed_chunks = semantic_analyzer.compute_semantic_similarity(
                                            chunks, embedder, method='concepts'
                                        )
                                        method_used = "concepts"
                                    else:  # Syntactic
                                        embeddings = []
                                        for chunk in chunks:
                                            embedding = embedder.embed_query(chunk)
                                            embeddings.append(embedding)
                                        similarity_matrix = cosine_similarity(np.array(embeddings))
                                        processed_chunks = chunks
                                        method_used = "syntactic"
                            
                            # Create heatmap(s)
                            if similarity_method == "Compare Both" or show_comparison:
                                # Show comparison of methods
                                st.subheader("\ud83d\udd0d Similarity Method Comparison")
                                
                                # Create side-by-side heatmaps
                                fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                                
                                # Syntactic heatmap
                                syntactic_matrix = comparison_results['syntactic'][0]
                                mask = np.triu(np.ones_like(syntactic_matrix, dtype=bool))
                                
                                sns.heatmap(
                                    syntactic_matrix,
                                    annot=True,
                                    fmt='.3f',
                                    cmap='RdYlBu_r',
                                    center=0.5,
                                    square=True,
                                    mask=mask,
                                    cbar_kws={"shrink": .8},
                                    ax=ax1
                                )
                                ax1.set_title('Syntactic Similarity\\n(Structure-based)', pad=20)
                                
                                # Semantic heatmap  
                                semantic_matrix = comparison_results['concepts'][0]
                                
                                sns.heatmap(
                                    semantic_matrix,
                                    annot=True,
                                    fmt='.3f',
                                    cmap='RdYlBu_r',
                                    center=0.5,
                                    square=True,
                                    mask=mask,
                                    cbar_kws={"shrink": .8},
                                    ax=ax2
                                )
                                ax2.set_title('Semantic Similarity\\n(Meaning-based)', pad=20)
                                
                                # Set labels for both
                                chunk_labels = [f"Chunk {i+1}" for i in range(len(chunks))]
                                for ax in [ax1, ax2]:
                                    ax.set_xticklabels(chunk_labels, rotation=45)
                                    ax.set_yticklabels(chunk_labels, rotation=0)
                                
                                plt.tight_layout()
                                st.pyplot(fig_comp)
                                
                                # Analysis of differences
                                analysis = semantic_analyzer.analyze_similarity_differences(
                                    chunks, syntactic_matrix, semantic_matrix
                                )
                                insights = semantic_analyzer.generate_similarity_insights(chunks, analysis)
                                
                                st.write("**\ud83d\udd0d Method Comparison Insights:**")
                                for insight in insights:
                                    st.markdown(insight)
                                    
                                # Use semantic for main analysis
                                similarity_matrix = semantic_matrix
                                
                            else:
                                # Single heatmap
                                fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 6))
                                
                                mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))
                                sns.heatmap(
                                    similarity_matrix, 
                                    annot=True, 
                                    fmt='.3f',
                                    cmap='RdYlBu_r',
                                    center=0.5,
                                    square=True,
                                    mask=mask,
                                    cbar_kws={"shrink": .8},
                                    ax=ax_heatmap
                                )
                                
                                # Customize labels and title
                                chunk_labels = [f"Chunk {i+1}" for i in range(len(chunks))]
                                ax_heatmap.set_xticklabels(chunk_labels, rotation=45)
                                ax_heatmap.set_yticklabels(chunk_labels, rotation=0)
                                
                                if method_used == "concepts":
                                    ax_heatmap.set_title('Semantic Similarity Between Chunks\\n(Meaning-based analysis)', pad=20)
                                else:
                                    ax_heatmap.set_title('Syntactic Similarity Between Chunks\\n(Structure-based analysis)', pad=20)
                                
                                plt.tight_layout()
                                st.pyplot(fig_heatmap)
                            
                            # Analysis insights
                            st.write("**Heatmap Insights:**")
                            
                            # Find most similar chunks
                            max_sim = 0
                            most_similar_pair = (0, 1)
                            
                            for i in range(len(chunks)):
                                for j in range(i+1, len(chunks)):
                                    if similarity_matrix[i, j] > max_sim:
                                        max_sim = similarity_matrix[i, j]
                                        most_similar_pair = (i, j)
                            
                            # Show what method detected
                            method_emoji = "\ud83e\udde0" if method_used == "concepts" else "\ud83d\udd27"
                            method_name = "Semantic" if method_used == "concepts" else "Syntactic"
                            
                            st.write(f"• **Most similar chunks ({method_name}):** Chunk {most_similar_pair[0]+1} & Chunk {most_similar_pair[1]+1} (similarity: {max_sim:.3f})")
                            
                            # Find most different chunks
                            min_sim = 1.0
                            most_different_pair = (0, 1)
                            
                            for i in range(len(chunks)):
                                for j in range(i+1, len(chunks)):
                                    if similarity_matrix[i, j] < min_sim:
                                        min_sim = similarity_matrix[i, j]
                                        most_different_pair = (i, j)
                            
                            st.write(f"• **Most different chunks:** Chunk {most_different_pair[0]+1} & Chunk {most_different_pair[1]+1} (similarity: {min_sim:.3f})")
                            
                            # Semantic clustering insights
                            avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                            st.write(f"• **Average similarity:** {avg_similarity:.3f}")
                            
                            if avg_similarity > 0.8:
                                st.info("🔄 **High similarity detected** - Chunks are semantically similar, consider consolidating")
                            elif avg_similarity < 0.3:
                                st.warning("⚡ **Low similarity detected** - Chunks are very different, may cause attention conflicts")
                            else:
                                st.success("✅ **Good semantic balance** - Chunks have appropriate diversity")
                            
                            # Detailed similarity breakdown
                            with st.expander("📊 Detailed Similarity Matrix", expanded=False):
                                similarity_df = pd.DataFrame(
                                    similarity_matrix,
                                    index=[f"Chunk {i+1}" for i in range(len(chunks))],
                                    columns=[f"Chunk {i+1}" for i in range(len(chunks))]
                                )
                                st.dataframe(similarity_df.style.background_gradient(cmap='RdYlBu_r', vmin=0, vmax=1))
                                
                                # Show chunk contents for reference
                                st.write("**Chunk Contents:**")
                                for i, chunk in enumerate(chunks):
                                    st.write(f"**Chunk {i+1}:** {chunk[:100]}{'...' if len(chunk) > 100 else ''}")
                            
                            # Add cluster analysis section
                            st.subheader("🎭 Cluster Analysis")
                            st.write("Discover thematic groups in your system prompt to understand how it affects LLM behavior.")
                            
                            # Clustering controls
                            cluster_col1, cluster_col2, cluster_col3 = st.columns([2, 1, 1])
                            with cluster_col1:
                                clustering_method = st.selectbox(
                                    "Clustering Method",
                                    ["Auto-detect", "Hierarchical", "K-means"],
                                    index=0,
                                    help="Choose how to group similar chunks"
                                )
                            
                            with cluster_col2:
                                manual_clusters = st.checkbox("Manual cluster count", value=False)
                            
                            with cluster_col3:
                                if manual_clusters:
                                    n_clusters = st.slider("Number of clusters", 2, min(8, len(chunks)), 3)
                                else:
                                    n_clusters = None
                            
                            # Perform clustering analysis
                            with st.spinner("Analyzing thematic clusters..."):
                                clustering_results = semantic_analyzer.perform_clustering_analysis(
                                    chunks, 
                                    similarity_matrix, 
                                    method=clustering_method.lower().replace('-', ''),
                                    n_clusters=n_clusters
                                )
                            
                            # Show all 3 visualizations
                            cluster_tabs = st.tabs(["📊 Feature Overview", "🎯 Cluster Matrix", "🔥 Similarity with Boundaries"])
                            
                            # Tab 1: Feature Overview Dashboard
                            with cluster_tabs[0]:
                                st.write("**🎯 Your System Prompt Structure:**")
                                
                                analysis = clustering_results['cluster_analysis']
                                n_clusters_found = clustering_results['n_clusters']
                                silhouette = clustering_results['silhouette_score']
                                
                                # Overview metrics
                                overview_col1, overview_col2, overview_col3 = st.columns(3)
                                with overview_col1:
                                    st.metric("Themes Detected", n_clusters_found)
                                with overview_col2:
                                    st.metric("Quality Score", f"{silhouette:.2f}")
                                with overview_col3:
                                    st.metric("Total Chunks", len(chunks))
                                
                                # Feature breakdown
                                for cluster_id, cluster_info in analysis.items():
                                    theme = cluster_info['theme']
                                    chunk_indices = cluster_info['chunk_indices']
                                    cohesion = cluster_info['cohesion']
                                    
                                    with st.expander(f"🔹 **{theme}** ({len(chunk_indices)} chunks, cohesion: {cohesion:.2f})", expanded=True):
                                        for i, chunk_idx in enumerate(chunk_indices):
                                            chunk_preview = chunks[chunk_idx][:80] + "..." if len(chunks[chunk_idx]) > 80 else chunks[chunk_idx]
                                            st.write(f"**Chunk {chunk_idx+1}:** {chunk_preview}")
                                        
                                        # Show concepts for this cluster
                                        if cluster_info['concepts']:
                                            st.write("**Key Concepts:**")
                                            for concept_type, items in cluster_info['concepts'].items():
                                                if items:
                                                    st.write(f"- *{concept_type.title()}*: {', '.join(items)}")
                            
                            # Tab 2: Cluster Assignment Matrix
                            with cluster_tabs[1]:
                                st.write("**Binary assignment showing which chunks belong to each cluster:**")
                                
                                # Create assignment matrix
                                assignment_matrix = np.zeros((len(chunks), n_clusters_found))
                                for i, label in enumerate(clustering_results['labels']):
                                    assignment_matrix[i, label] = 1
                                
                                # Plot assignment matrix
                                fig_assign, ax_assign = plt.subplots(figsize=(max(6, n_clusters_found * 1.5), max(4, len(chunks) * 0.4)))
                                
                                im = ax_assign.imshow(assignment_matrix, cmap='RdYlBu_r', aspect='auto')
                                
                                # Add text annotations
                                for i in range(len(chunks)):
                                    for j in range(n_clusters_found):
                                        text = "●" if assignment_matrix[i, j] == 1 else "○"
                                        ax_assign.text(j, i, text, ha="center", va="center", fontsize=14, 
                                                     color="white" if assignment_matrix[i, j] == 1 else "gray")
                                
                                # Set labels
                                cluster_themes = [analysis[i]['theme'] for i in range(n_clusters_found)]
                                ax_assign.set_xticks(range(n_clusters_found))
                                ax_assign.set_xticklabels([f"Cluster {i+1}\n{theme[:20]}..." if len(theme) > 20 else f"Cluster {i+1}\n{theme}" 
                                                          for i, theme in enumerate(cluster_themes)], rotation=45, ha='right')
                                
                                chunk_labels = [f"Chunk {i+1}" for i in range(len(chunks))]
                                ax_assign.set_yticks(range(len(chunks)))
                                ax_assign.set_yticklabels(chunk_labels)
                                
                                ax_assign.set_title('Chunk-to-Cluster Assignment Matrix', pad=20)
                                ax_assign.set_xlabel('Thematic Clusters')
                                ax_assign.set_ylabel('System Prompt Chunks')
                                
                                plt.tight_layout()
                                st.pyplot(fig_assign)
                            
                            # Tab 3: Similarity Matrix with Cluster Boundaries
                            with cluster_tabs[2]:
                                st.write("**Similarity matrix reorganized by clusters to show thematic groupings:**")
                                
                                # Reorder similarity matrix by clusters
                                cluster_order = []
                                for cluster_id in range(n_clusters_found):
                                    cluster_indices = clustering_results['clusters'][cluster_id]
                                    cluster_order.extend(cluster_indices)
                                
                                reordered_matrix = similarity_matrix[np.ix_(cluster_order, cluster_order)]
                                
                                # Plot reordered similarity matrix with boundaries
                                fig_bound, ax_bound = plt.subplots(figsize=(10, 8))
                                
                                mask = np.triu(np.ones_like(reordered_matrix, dtype=bool))
                                sns.heatmap(
                                    reordered_matrix,
                                    annot=True,
                                    fmt='.2f',
                                    cmap='RdYlBu_r',
                                    center=0.5,
                                    square=True,
                                    mask=mask,
                                    cbar_kws={"shrink": .8},
                                    ax=ax_bound
                                )
                                
                                # Add cluster boundaries
                                current_pos = 0
                                for cluster_id in range(n_clusters_found - 1):  # Don't draw line after last cluster
                                    cluster_size = len(clustering_results['clusters'][cluster_id])
                                    current_pos += cluster_size
                                    # Draw vertical and horizontal lines to separate clusters
                                    ax_bound.axvline(x=current_pos, color='red', linewidth=2, alpha=0.7)
                                    ax_bound.axhline(y=current_pos, color='red', linewidth=2, alpha=0.7)
                                
                                # Create custom labels showing cluster membership
                                reordered_labels = []
                                for cluster_id in range(n_clusters_found):
                                    cluster_indices = clustering_results['clusters'][cluster_id]
                                    theme = analysis[cluster_id]['theme']
                                    for idx in cluster_indices:
                                        reordered_labels.append(f"C{cluster_id+1}-{idx+1}")  # Cluster1-Chunk1 format
                                
                                ax_bound.set_xticklabels(reordered_labels, rotation=45, ha='right')
                                ax_bound.set_yticklabels(reordered_labels, rotation=0)
                                ax_bound.set_title('Similarity Matrix Grouped by Clusters\n(Red lines show cluster boundaries)', pad=20)
                                
                                plt.tight_layout()
                                st.pyplot(fig_bound)
                                
                                # Show cluster legend
                                st.write("**Cluster Legend:**")
                                for cluster_id, cluster_info in analysis.items():
                                    theme = cluster_info['theme']
                                    size = cluster_info['size']
                                    st.write(f"• **C{cluster_id+1}**: {theme} ({size} chunks)")
                            
                            # Generate cluster insights
                            cluster_insights = semantic_analyzer.generate_cluster_insights(chunks, clustering_results)
                            st.write("**🎯 Clustering Insights & LLM Impact:**")
                            for insight in cluster_insights:
                                st.markdown(insight)
                            
                            # Add cognitive load analysis
                            st.markdown("---")
                            st.subheader("🧠 Cognitive Load Analysis")
                            st.write("Predicts LLM output degradation risk based on prompt complexity.")
                            
                            # Generate cognitive load analysis
                            from semantic_similarity import CognitiveLoadAnalyzer
                            cognitive_load_analyzer = CognitiveLoadAnalyzer()
                            load_result = cognitive_load_analyzer.analyze_cognitive_load(
                                chunks, clustering_results, similarity_matrix, prediction.token_importance
                            )

                            # Main load dashboard
                            load_col1, load_col2, load_col3 = st.columns([2, 1, 1])

                            with load_col1:
                                # Load meter with color coding
                                load_color = "🔴" if load_result.overall_load >= 80 else "⚠️" if load_result.overall_load >= 60 else "🟡" if load_result.overall_load >= 40 else "✅"
                                st.metric("Overall Cognitive Load", f"{load_result.overall_load:.0f}/100", help="Measures prompt complexity and degradation risk")
                                
                            with load_col2:
                                st.metric("Risk Level", f"{load_color} {load_result.risk_level}")
                                
                            with load_col3:
                                # Dominant load factor
                                max_factor = max(load_result.breakdown.keys(), key=lambda k: load_result.breakdown[k])
                                st.metric("Primary Issue", max_factor.replace('_', ' ').title())

                            # Detailed load breakdown
                            st.write("**Load Breakdown:**")
                            breakdown_cols = st.columns(len(load_result.breakdown))

                            for i, (factor, score) in enumerate(load_result.breakdown.items()):
                                with breakdown_cols[i]:
                                    factor_name = factor.replace('_', ' ').title()
                                    bar_length = int(score / 10)
                                    progress_bar = "█" * bar_length + "░" * (10 - bar_length)
                                    st.write(f"**{factor_name}**")
                                    st.write(f"`{progress_bar}` {score:.0f}")

                            # Risk warnings and conflicts
                            if load_result.conflicts:
                                st.write("**🚨 Conflicts Detected:**")
                                for conflict in load_result.conflicts[:3]:  # Show top 3
                                    st.error(conflict)

                            # Optimization recommendations
                            if load_result.recommendations:
                                st.write("**💡 Optimization Recommendations:**")
                                for i, rec in enumerate(load_result.recommendations, 1):
                                    st.info(f"**{i}.** {rec}")

                            # Cognitive load heatmap
                            if len(clustering_results['clusters']) > 1:
                                st.write("**🔥 Cognitive Interference Heatmap:**")
                                
                                fig_load, ax_load = plt.subplots(figsize=(8, 6))
                                
                                sns.heatmap(
                                    load_result.load_heatmap,
                                    annot=True,
                                    fmt='.0f',
                                    cmap='Reds',
                                    square=True,
                                    cbar_kws={"shrink": .8},
                                    ax=ax_load
                                )
                                
                                cluster_themes = [analysis[i]['theme'][:15] + "..." if len(analysis[i]['theme']) > 15 
                                                else analysis[i]['theme'] for i in range(len(clustering_results['clusters']))]
                                
                                ax_load.set_xticklabels(cluster_themes, rotation=45, ha='right')
                                ax_load.set_yticklabels(cluster_themes, rotation=0)
                                ax_load.set_title('Cognitive Interference Between Clusters\n(Higher values = more mental switching cost)', pad=20)
                                
                                plt.tight_layout()
                                st.pyplot(fig_load)
                                
                                # Heatmap insights
                                st.write("**Interference Insights:**")
                                max_interference = np.max(load_result.load_heatmap[load_result.load_heatmap < 100])  # Exclude diagonal
                                if max_interference > 70:
                                    st.warning(f"⚠️ High interference detected ({max_interference:.0f}%) - may cause context switching confusion")
                                elif max_interference > 40:
                                    st.info(f"📊 Moderate interference ({max_interference:.0f}%) - manageable complexity")
                                else:
                                    st.success(f"✅ Low interference ({max_interference:.0f}%) - smooth cognitive flow")
                        
                        except Exception as e:
                            st.error(f"Error creating embeddings heatmap: {str(e)}")
                            st.write("💡 **Note:** Embeddings heatmap requires OpenAI API access. ")
                            st.write("The heatmap shows semantic similarity between chunks based on their embeddings.")
                    
                    else:
                        st.info("🔑 **Embeddings heatmap requires OpenAI API key**")
                        st.write("The heatmap shows how semantically similar different chunks are to each other.")
                        st.write("• **Red areas**: High similarity (chunks discuss similar topics)")
                        st.write("• **Blue areas**: Low similarity (chunks discuss different topics)")
                        st.write("• **Diagonal**: Always 1.0 (chunks compared to themselves)")
                        
                        # Create a demo heatmap with mock data
                        st.write("**Demo Heatmap (Mock Data):**")
                        demo_matrix = np.random.rand(len(chunks), len(chunks))
                        # Make it symmetric
                        demo_matrix = (demo_matrix + demo_matrix.T) / 2
                        # Set diagonal to 1
                        np.fill_diagonal(demo_matrix, 1.0)
                        
                        fig_demo, ax_demo = plt.subplots(figsize=(6, 5))
                        mask = np.triu(np.ones_like(demo_matrix, dtype=bool))
                        sns.heatmap(
                            demo_matrix,
                            annot=True,
                            fmt='.3f',
                            cmap='RdYlBu_r',
                            center=0.5,
                            square=True,
                            mask=mask,
                            cbar_kws={"shrink": .8},
                            ax=ax_demo
                        )
                        
                        chunk_labels = [f"Chunk {i+1}" for i in range(len(chunks))]
                        ax_demo.set_xticklabels(chunk_labels, rotation=45)
                        ax_demo.set_yticklabels(chunk_labels, rotation=0)
                        ax_demo.set_title('Demo: Chunk Similarity Heatmap', pad=20)
                        
                        plt.tight_layout()
                        st.pyplot(fig_demo)
            
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            with st.expander("Debug Information"):
                import traceback
                st.text(traceback.format_exc())
        
        # OpenAI Testing Section
        if test_with_openai and api_key:
            st.markdown("---")
            st.markdown("---")
            st.subheader("🤖 OpenAI Generation Test")
            
            # Show which system prompt will be used
            with st.expander("📄 System Prompt Being Tested", expanded=False):
                st.code(prompt, language="text")
                st.caption(f"This system prompt ({len(prompt)} characters) will be sent to {model_name}")
            
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                
                st.info(f"Testing how your system prompt affects {model_name} responses...")
                
                # Parse test questions
                questions = [q.strip() for q in test_questions.split('\n') if q.strip()]
                
                # Test each question
                for i, question in enumerate(questions[:3], 1):  # Limit to 3 questions to control costs
                    st.markdown(f"### 📝 Test {i}: {question}")
                    try:
                        # Make API call with the analyzed system prompt
                        with st.spinner(f"Generating response for: {question}"):
                            # Debug info
                            st.caption(f"Sending to {model_name} with system prompt ({len(prompt)} chars)")
                            
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=[
                                    {"role": "system", "content": prompt},
                                    {"role": "user", "content": question}
                                ],
                                max_tokens=150,
                                temperature=0.7
                            )
                        
                        # Display response
                        answer = response.choices[0].message.content
                        st.write("**Response:**")
                        st.write(answer)
                        
                        # Show tokens used for cost tracking
                        if hasattr(response, 'usage'):
                            st.caption(f"Tokens used: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")
                        
                        # Analyze response against attention predictions
                        st.write("**Attention Analysis:**")
                        
                        # Check if response aligns with critical chunks
                        response_analysis = []
                        for idx in prediction.critical_tokens[:2]:  # Check top 2 critical chunks
                            if idx < len(chunks):
                                chunk_preview = chunks[idx][:50].replace('\n', ' ')
                                chunk_importance = prediction.token_importance[idx]
                                
                                # Simple keyword matching to see if response reflects chunk content
                                chunk_keywords = set(chunk_preview.lower().split())
                                response_keywords = set(answer.lower().split())
                                overlap = len(chunk_keywords.intersection(response_keywords))
                                
                                response_analysis.append({
                                    "chunk": chunk_preview,
                                    "importance": chunk_importance,
                                    "keyword_overlap": overlap,
                                    "alignment": "High" if overlap > 2 else "Medium" if overlap > 0 else "Low"
                                })
                        
                        # Display alignment analysis
                        if response_analysis:
                            analysis_df = pd.DataFrame(response_analysis)
                            st.dataframe(analysis_df, width="stretch")
                            
                            # Summary insight
                            high_alignment = sum(1 for a in response_analysis if a["alignment"] == "High")
                            if high_alignment > 0:
                                st.success(f"✅ Response aligns with {high_alignment} high-importance chunks - attention predictions validated!")
                            else:
                                st.warning("⚠️ Response may not fully reflect highest-importance chunks - consider prompt revision")
                        
                    except Exception as api_error:
                        st.error(f"OpenAI API Error: {str(api_error)}")
                        if "quota" in str(api_error).lower():
                            st.info("💡 Tip: Check your OpenAI account billing and usage limits")
                        elif "api key" in str(api_error).lower():
                            st.info("💡 Tip: Verify your API key is valid and has the correct permissions")
                
                # Cost estimation and system prompt confirmation
                total_questions = len(questions[:3])
                prompt_length = len(prompt)
                estimated_prompt_tokens = prompt_length // 4  # Rough estimate: 4 chars per token
                
                st.info(f"💰 **Estimated Cost:** ~$0.01-0.05 for testing {total_questions} questions with {model_name}")
                st.info(f"📄 **System Prompt:** {prompt_length} characters (~{estimated_prompt_tokens} tokens) sent with each request")
                
                # Prompt optimization suggestions based on results
                st.subheader("🎯 Optimization Insights")
                st.write("**How to use these results:**")
                st.write("1. **High Alignment**: Your attention predictions match actual outputs - good prompt structure!")
                st.write("2. **Low Alignment**: Consider reordering or emphasizing critical chunks")
                st.write("3. **Unexpected Responses**: May indicate attention competition between chunks")
                st.write("4. **System Prompt Impact**: Compare responses to see how your prompt guides behavior")
                
                # Quick validation test
                st.write("\n**🔍 Quick Validation:**")
                st.write(f"- System prompt length: {len(prompt)} characters")
                st.write(f"- Number of chunks analyzed: {len(chunks)}")
                st.write(f"- Questions tested: {total_questions}")
                st.write(f"- Model used: {model_name}")
                
            except ImportError:
                st.error("OpenAI library not installed. Run: `pip install openai`")
            except Exception as e:
                st.error(f"Error setting up OpenAI client: {str(e)}")
                st.write("**Debugging info:**")
                st.write(f"- API key provided: {'Yes' if api_key else 'No'}")
                st.write(f"- API key length: {len(api_key) if api_key else 0} characters")
                st.write(f"- Selected model: {model_name}")
                st.write(f"- System prompt length: {len(prompt)} characters")

# Footer
st.markdown("---")
st.markdown("---")
# Spacer to ensure content is visible
st.markdown("<div style='margin-bottom: 100px;'></div>", unsafe_allow_html=True)

with st.expander("📚 About This Demo", expanded=False):
    st.markdown("""
### 🎯 What This Demo Shows

This attention flow analysis predicts how transformer models will focus their attention on different parts of your prompt **without running the actual model**. 

**Key Features:**
- 🧮 **Mathematical Analysis**: Uses graph theory and information theory
- ⚡ **Fast & Cost-Free**: No API calls required
- 📊 **Actionable Insights**: Specific optimization recommendations
- 🔍 **Token-Level Detail**: See exactly which words get attention

**Business Value:**
- Optimize prompts before deploying to production
- Reduce LLM API costs through better prompt engineering
- Identify attention conflicts and bottlenecks
- Improve prompt effectiveness systematically

**🔄 Feedback Loop:**
- Analyze attention flow predictions
- Test with real OpenAI generations  
- Compare predictions vs actual behavior
- Optimize prompts based on results

Built with ❤️ for better prompt engineering!
""")
