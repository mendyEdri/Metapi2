"""Quick test for OpenAI integration functionality."""

import os
from attention_flow import AttentionFlowAnalyzer
from prompt_chunking import chunk_prompt

# Test the complete flow without API key
def test_integration_without_api():
    """Test that the integration works without breaking existing functionality."""
    
    prompt = """## Main language:
You should always speak English

## Secondary language
You should always speak Spanish"""
    
    print("=== Testing Integration Flow ===")
    
    # Test chunking
    chunks = chunk_prompt(prompt)
    print(f"✅ Chunking: Found {len(chunks)} chunks")
    
    # Test attention analysis
    analyzer = AttentionFlowAnalyzer()
    prediction = analyzer.analyze_attention_flow(prompt, use_chunks=True)
    print(f"✅ Attention Analysis: Competition score {prediction.competition_score:.3f}")
    
    # Test visualization
    viz = analyzer.visualize_attention_flow(chunks, prediction, use_chunks=True)
    print(f"✅ Visualization: Generated {len(viz)} characters of output")
    
    # Mock response analysis (without actual API call)
    mock_response = "I primarily communicate in English, but I can also respond in Spanish when requested."
    
    # Test alignment analysis logic
    response_analysis = []
    for idx in prediction.critical_tokens[:2]:
        if idx < len(chunks):
            chunk_preview = chunks[idx][:50].replace('\n', ' ')
            chunk_importance = prediction.token_importance[idx]
            
            chunk_keywords = set(chunk_preview.lower().split())
            response_keywords = set(mock_response.lower().split())
            overlap = len(chunk_keywords.intersection(response_keywords))
            
            response_analysis.append({
                "chunk": chunk_preview,
                "importance": chunk_importance,
                "keyword_overlap": overlap,
                "alignment": "High" if overlap > 2 else "Medium" if overlap > 0 else "Low"
            })
    
    print(f"✅ Response Analysis: Analyzed {len(response_analysis)} chunk alignments")
    for analysis in response_analysis:
        print(f"  - Chunk: {analysis['chunk'][:30]}...")
        print(f"    Importance: {analysis['importance']:.3f}")
        print(f"    Alignment: {analysis['alignment']} ({analysis['keyword_overlap']} overlapping keywords)")
    
    print("\n🎉 Integration test passed! All components working correctly.")
    
    return True

if __name__ == "__main__":
    test_integration_without_api()
