#!/usr/bin/env python3
"""
Test script for cluster analysis functionality.
Tests the Hebrew/Spanish example to validate clustering works correctly.
"""

import numpy as np
from semantic_similarity import SemanticSimilarityAnalyzer

# Mock embedder for testing without API key
class MockEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        # Simple mock: assign vectors based on keywords/concepts
        if "hebrew" in text.lower():
            return np.array([0.9, 0.1, 0.1, 0.0, 0.0])
        elif "spanish" in text.lower():
            return np.array([0.1, 0.9, 0.1, 0.0, 0.0])
        elif "nice" in text.lower() or "helpful" in text.lower():
            return np.array([0.0, 0.0, 0.1, 0.9, 0.1])
        elif "empty" in text.lower(): # For empty semantic content
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5]) # Neutral for other words

print("🧪 TESTING CLUSTER ANALYSIS")
print("=" * 50)

# User's Hebrew/Spanish example
chunks = [
    "You might speaks hebrew",
    "from some time to time answer in hebrew",
    "You might speak spanish",
    "Be nice and helpful"
]

print("\n📝 Test Chunks:")
for i, chunk in enumerate(chunks):
    print(f"  {i+1}. '{chunk}'")

# Initialize analyzer and mock embedder
analyzer = SemanticSimilarityAnalyzer()
mock_embedder = MockEmbedder()

print("\n🧠 SEMANTIC SIMILARITY:")
print("-" * 30)

# Compute semantic similarity
similarity_matrix, processed_chunks = analyzer.compute_semantic_similarity(
    chunks, mock_embedder, method='concepts'
)

print("Similarity Matrix:")
print(similarity_matrix)

print("\n🎭 CLUSTER ANALYSIS:")
print("-" * 30)

# Test all clustering methods
methods = ['hierarchical', 'kmeans', 'auto']

for method in methods:
    print(f"\n**Testing {method.upper()} clustering:**")
    
    try:
        clustering_results = analyzer.perform_clustering_analysis(
            chunks, similarity_matrix, method=method
        )
        
        print(f"  Clusters found: {clustering_results['n_clusters']}")
        print(f"  Silhouette score: {clustering_results['silhouette_score']:.3f}")
        print(f"  Method used: {clustering_results['method_used']}")
        
        print("  Cluster assignments:")
        for i, label in enumerate(clustering_results['labels']):
            print(f"    Chunk {i+1}: Cluster {label+1}")
        
        print("  Cluster analysis:")
        for cluster_id, info in clustering_results['cluster_analysis'].items():
            print(f"    Cluster {cluster_id+1}: {info['theme']} ({info['size']} chunks, cohesion: {info['cohesion']:.3f})")
        
        # Generate insights
        insights = analyzer.generate_cluster_insights(chunks, clustering_results)
        print("  Key insights:")
        for insight in insights[:3]:  # Show first 3 insights
            print(f"    - {insight}")
            
    except Exception as e:
        print(f"  Error with {method}: {e}")

print("\n🎯 EXPECTED RESULTS:")
print("-" * 30)
print("For the Hebrew/Spanish example, we should see:")
print("- 2-3 clusters detected")
print("- Hebrew chunks (1 & 2) grouped together")
print("- Spanish chunk (3) in separate cluster")  
print("- Behavior chunk (4) may be separate or grouped")
print("- Clear thematic separation with good insights")

print("\n✅ CLUSTER ANALYSIS TEST COMPLETE!")
print("If clustering is working correctly, Hebrew chunks should be in the same cluster.")
