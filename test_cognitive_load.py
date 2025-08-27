#!/usr/bin/env python3
"""
Test script for cognitive load analyzer functionality.
Tests various scenarios to validate load measurement accuracy.
"""

import numpy as np
from semantic_similarity import SemanticSimilarityAnalyzer, CognitiveLoadAnalyzer

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
        elif "always" in text.lower():
            return np.array([0.8, 0.0, 0.0, 0.0, 0.2])
        elif "never" in text.lower():
            return np.array([0.0, 0.8, 0.0, 0.0, 0.2])
        else:
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5])

print("🧠 TESTING COGNITIVE LOAD ANALYZER")
print("=" * 50)

# Test cases with expected load levels
test_cases = [
    {
        "name": "Low Load (Consistent)",
        "chunks": ["Be helpful", "Be nice", "Be kind"],
        "expected_load_range": (10, 30),
        "expected_risk": "Low"
    },
    {
        "name": "Medium Load (Mixed Domains)",
        "chunks": ["Speak Spanish", "Be professional", "Calculate math"],
        "expected_load_range": (40, 65),
        "expected_risk": "Medium"
    },
    {
        "name": "High Load (Conflicts)",
        "chunks": [
            "Always respond in Spanish",
            "Never respond in Spanish", 
            "Be helpful",
            "Calculate complex equations"
        ],
        "expected_load_range": (65, 90),
        "expected_risk": "High"
    }
]

analyzer = SemanticSimilarityAnalyzer()
cognitive_analyzer = CognitiveLoadAnalyzer()
mock_embedder = MockEmbedder()

for i, test_case in enumerate(test_cases, 1):
    print(f"\n🧪 TEST {i}: {test_case['name']}")
    print("-" * 40)
    
    chunks = test_case['chunks']
    print(f"Chunks: {chunks}")
    
    try:
        # Compute similarity matrix
        similarity_matrix, _ = analyzer.compute_semantic_similarity(chunks, mock_embedder, method='concepts')
        
        # Perform clustering
        clustering_results = analyzer.perform_clustering_analysis(chunks, similarity_matrix, method='auto')
        
        # Analyze cognitive load
        mock_attention_scores = np.random.rand(len(chunks))  # Mock attention scores
        load_result = cognitive_analyzer.analyze_cognitive_load(
            chunks, clustering_results, similarity_matrix, mock_attention_scores
        )
        
        # Display results
        print(f"Overall Load: {load_result.overall_load:.1f}/100")
        print(f"Risk Level: {load_result.risk_level}")
        print(f"Breakdown:")
        for factor, score in load_result.breakdown.items():
            print(f"  - {factor.replace('_', ' ').title()}: {score:.1f}")
        
        if load_result.conflicts:
            print(f"Conflicts: {len(load_result.conflicts)}")
            for conflict in load_result.conflicts:
                print(f"  - {conflict}")
        
        # Validate results
        expected_min, expected_max = test_case['expected_load_range']
        expected_risk = test_case['expected_risk']
        
        if expected_min <= load_result.overall_load <= expected_max:
            print(f"✅ Load in expected range [{expected_min}-{expected_max}]")
        else:
            print(f"❌ Load {load_result.overall_load:.1f} outside expected range [{expected_min}-{expected_max}]")
            
        if load_result.risk_level == expected_risk:
            print(f"✅ Risk level matches expected: {expected_risk}")
        else:
            print(f"❌ Risk level {load_result.risk_level} != expected {expected_risk}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

print("\n🎯 EXPECTED BEHAVIOR:")
print("-" * 30)
print("Low Load: Consistent, similar concepts → Low cognitive burden")
print("Medium Load: Mixed domains but no conflicts → Manageable complexity")  
print("High Load: Direct contradictions → High degradation risk")

print("\n✅ COGNITIVE LOAD ANALYZER TESTS COMPLETE!")
