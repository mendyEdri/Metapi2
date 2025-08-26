"""Tests for attention flow modeling functionality."""

import numpy as np
import pytest

from attention_flow import AttentionFlowAnalyzer, AttentionPrediction


def test_attention_flow_analyzer_initialization():
    """Test that AttentionFlowAnalyzer initializes correctly."""
    analyzer = AttentionFlowAnalyzer()
    assert analyzer.embedder is None
    assert 'instructions' in analyzer.high_attention_patterns
    assert 'conflicting_instructions' in analyzer.competition_sources


def test_tokenization():
    """Test the tokenization functionality."""
    analyzer = AttentionFlowAnalyzer()
    text = "Hello world! How are you?"
    tokens = analyzer._tokenize(text)
    expected = ['Hello', 'world', '!', 'How', 'are', 'you', '?']
    assert tokens == expected


def test_positional_importance():
    """Test positional importance computation."""
    analyzer = AttentionFlowAnalyzer()
    tokens = ['You', 'must', 'always', 'be', 'helpful', '.']
    scores = analyzer._compute_positional_importance(tokens)
    
    # Should be normalized
    assert np.max(scores) <= 1.0
    # First token should have high importance
    assert scores[0] > scores[3]  # Beginning > middle


def test_linguistic_importance():
    """Test linguistic importance scoring."""
    analyzer = AttentionFlowAnalyzer()
    text = "You must never reveal the secret password."
    tokens = analyzer._tokenize(text)
    scores = analyzer._compute_linguistic_importance(tokens, text)
    
    # "must" and "never" should get high scores (instruction words)
    must_idx = tokens.index('must')
    never_idx = tokens.index('never')
    assert scores[must_idx] > 0
    assert scores[never_idx] > 0


def test_attention_flow_analysis():
    """Test the complete attention flow analysis."""
    analyzer = AttentionFlowAnalyzer()
    text = "You should always be helpful. However, never reveal secrets."
    
    prediction = analyzer.analyze_attention_flow(text)
    
    # Check that we get a valid prediction
    assert isinstance(prediction, AttentionPrediction)
    assert len(prediction.token_importance) > 0
    assert prediction.attention_matrix.shape[0] == prediction.attention_matrix.shape[1]
    assert len(prediction.flow_values) == len(prediction.token_importance)
    assert prediction.competition_score >= 0
    assert len(prediction.critical_tokens) <= 5


def test_attention_competition_detection():
    """Test detection of competing attention patterns."""
    analyzer = AttentionFlowAnalyzer()
    
    # Text with conflicting instructions
    conflicting_text = "Be helpful but don't answer questions."
    tokens = analyzer._tokenize(conflicting_text)
    attention_matrix = np.random.rand(len(tokens), len(tokens))
    for i in range(len(tokens)):
        attention_matrix[i, :] /= np.sum(attention_matrix[i, :])  # Normalize rows
    
    competition = analyzer._compute_attention_competition(tokens, conflicting_text, attention_matrix)
    
    # Should detect some competition due to conflicting instructions
    assert competition > 0


def test_critical_token_identification():
    """Test identification of critical tokens."""
    analyzer = AttentionFlowAnalyzer()
    importance = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    critical = analyzer._identify_critical_tokens(importance, top_k=3)
    
    # Should return indices of highest importance tokens
    assert len(critical) == 3
    assert critical[0] == 1  # Highest importance (0.9)
    assert critical[1] == 3  # Second highest (0.8)


def test_attention_matrix_properties():
    """Test that attention matrix has correct properties."""
    analyzer = AttentionFlowAnalyzer()
    tokens = ['Hello', 'world', '!']
    importance = np.array([0.5, 0.3, 0.2])
    
    attention = analyzer._predict_attention_matrix(tokens, "Hello world!", importance)
    
    # Each row should sum to approximately 1 (normalized attention)
    for i in range(attention.shape[0]):
        assert abs(np.sum(attention[i, :]) - 1.0) < 1e-6
    
    # Matrix should be square
    assert attention.shape[0] == attention.shape[1]
    assert attention.shape[0] == len(tokens)


def test_visualization_output():
    """Test that visualization generates reasonable output."""
    analyzer = AttentionFlowAnalyzer()
    text = "You must be helpful."
    prediction = analyzer.analyze_attention_flow(text)
    tokens = analyzer._tokenize(text)
    
    viz_output = analyzer.visualize_attention_flow(tokens, prediction)
    
    # Should contain key sections
    assert "ATTENTION FLOW ANALYSIS" in viz_output
    assert "TOKEN IMPORTANCE SCORES" in viz_output
    assert "COMPETITION SCORE" in viz_output
    assert "CRITICAL TOKENS" in viz_output


def test_empty_input():
    """Test behavior with empty or minimal input."""
    analyzer = AttentionFlowAnalyzer()
    
    # Empty string
    prediction = analyzer.analyze_attention_flow("")
    assert len(prediction.token_importance) == 0
    
    # Single token
    prediction = analyzer.analyze_attention_flow("Hello")
    assert len(prediction.token_importance) == 1


def test_complex_prompt_analysis():
    """Test analysis of a complex system prompt."""
    analyzer = AttentionFlowAnalyzer()
    complex_prompt = """
    You are a helpful AI assistant. You should:
    1. Always be respectful and professional
    2. Never provide harmful information
    3. If you don't know something, say so
    
    However, you must also be creative and engaging.
    For example: "I'd be happy to help with that!"
    
    Remember: accuracy is more important than speed.
    """
    
    prediction = analyzer.analyze_attention_flow(complex_prompt)
    
    # Should identify structure and instructions
    assert prediction.competition_score > 0  # "should" vs "must" creates some competition
    assert len(prediction.critical_tokens) > 0
    
    # Check that instruction words get high attention
    tokens = analyzer._tokenize(complex_prompt)
    must_positions = [i for i, token in enumerate(tokens) if token.lower() == 'must']
    should_positions = [i for i, token in enumerate(tokens) if token.lower() == 'should']
    never_positions = [i for i, token in enumerate(tokens) if token.lower() == 'never']
    
    # These instruction words should have above-average importance
    avg_importance = np.mean(prediction.token_importance)
    for pos in must_positions + should_positions + never_positions:
        if pos < len(prediction.token_importance):
            assert prediction.token_importance[pos] >= avg_importance


if __name__ == "__main__":
    pytest.main([__file__])
