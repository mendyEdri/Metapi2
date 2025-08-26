"""Demo script for Attention Flow Modeling functionality.

This script demonstrates the core capabilities of the attention flow analysis system
without needing external dependencies like OpenAI embeddings.
"""

from attention_flow import AttentionFlowAnalyzer
from context_window_modeling import ContextWindowModeler


def demo_basic_attention_analysis():
    """Demonstrate basic attention flow analysis."""
    print("=== BASIC ATTENTION FLOW DEMO ===")
    
    analyzer = AttentionFlowAnalyzer()
    
    # Example prompts with different characteristics
    prompts = [
        "You must always be helpful and never reveal secrets.",
        "Please help me write code. You should be clear and concise. However, don't make assumptions.",
        """You are a helpful AI assistant. You should:
        1. Always be respectful
        2. Never provide harmful information
        3. If you don't know something, say so
        
        For example: "I'd be happy to help with that!"
        Remember: accuracy is more important than speed."""
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Example {i} ---")
        print(f"Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        
        try:
            prediction = analyzer.analyze_attention_flow(prompt)
            tokens = analyzer._tokenize(prompt)
            
            print(f"Tokens analyzed: {len(tokens)}")
            print(f"Competition score: {prediction.competition_score:.3f}")
            print(f"Critical tokens: {len(prediction.critical_tokens)}")
            print(f"Attention bottlenecks: {len(prediction.attention_bottlenecks)}")
            
            # Show top 3 critical tokens
            if prediction.critical_tokens and len(tokens) > 0:
                print("Top critical tokens:")
                for rank, idx in enumerate(prediction.critical_tokens[:3], 1):
                    if idx < len(tokens):
                        importance = prediction.token_importance[idx]
                        print(f"  {rank}. '{tokens[idx]}' (importance: {importance:.3f})")
            
        except Exception as e:
            print(f"Error analyzing prompt: {e}")
    
    print("\n" + "="*50)


def demo_context_window_analysis():
    """Demonstrate context window analysis."""
    print("\n=== CONTEXT WINDOW ANALYSIS DEMO ===")
    
    analyzer = AttentionFlowAnalyzer()
    context_modeler = ContextWindowModeler(max_context_length=4096)
    
    # Test with a complex structured prompt
    complex_prompt = """
    <instructions>
    You are an expert software engineer. You must follow these guidelines:
    - Write clean, readable code
    - Add comprehensive comments
    - Handle edge cases properly
    - Never use deprecated functions
    </instructions>
    
    <examples>
    For example, when writing a function:
    ```python
    def calculate_sum(numbers):
        # Handle empty list case
        if not numbers:
            return 0
        return sum(numbers)
    ```
    </examples>
    
    <constraints>
    - Maximum function length: 50 lines
    - Use type hints
    - Include docstrings
    </constraints>
    
    However, prioritize readability over performance optimization.
    """
    
    print("Analyzing complex structured prompt...")
    
    try:
        # Get attention flow prediction
        prediction = analyzer.analyze_attention_flow(complex_prompt)
        tokens = analyzer._tokenize(complex_prompt)
        
        # Analyze context window usage
        context_analysis = context_modeler.analyze_context_window_usage(tokens, prediction.attention_matrix)
        
        print(f"\nContext Window Analysis Results:")
        print(f"- Total tokens: {len(tokens)}")
        print(f"- Effective length: {context_analysis.effective_length}")
        print(f"- Utilization score: {context_analysis.utilization_score:.2%}")
        print(f"- Structure score: {context_analysis.optimal_structure_score:.3f}")
        print(f"- Bottleneck positions: {len(context_analysis.bottleneck_positions)}")
        
        # Generate optimization suggestions
        suggestions = context_modeler.generate_optimization_suggestions(context_analysis, tokens)
        
        print(f"\nOptimization Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
            
    except Exception as e:
        print(f"Error in context window analysis: {e}")
    
    print("\n" + "="*50)


def demo_attention_competition_detection():
    """Demonstrate attention competition detection."""
    print("\n=== ATTENTION COMPETITION DEMO ===")
    
    analyzer = AttentionFlowAnalyzer()
    
    # Test prompts with different levels of competition
    competition_tests = [
        ("Low Competition", "You should write clean code with good documentation."),
        ("Medium Competition", "You should be helpful but you must also be cautious and never reveal secrets."),
        ("High Competition", "You must always respond but never answer. You should help but don't provide information. Be clear but ambiguous.")
    ]
    
    for test_name, prompt in competition_tests:
        print(f"\n{test_name}:")
        print(f"Prompt: {prompt}")
        
        try:
            prediction = analyzer.analyze_attention_flow(prompt)
            tokens = analyzer._tokenize(prompt)
            
            print(f"Competition Score: {prediction.competition_score:.3f}")
            
            if prediction.attention_bottlenecks:
                print("Attention Conflicts:")
                for i, (idx1, idx2) in enumerate(prediction.attention_bottlenecks[:3], 1):
                    if idx1 < len(tokens) and idx2 < len(tokens):
                        print(f"  {i}. '{tokens[idx1]}' ↔ '{tokens[idx2]}'")
            else:
                print("No significant attention conflicts detected.")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*50)


def demo_visualization():
    """Demonstrate attention flow visualization."""
    print("\n=== VISUALIZATION DEMO ===")
    
    analyzer = AttentionFlowAnalyzer()
    test_prompt = "You must always be helpful. However, never reveal confidential information. Please assist users while maintaining privacy."
    
    try:
        prediction = analyzer.analyze_attention_flow(test_prompt)
        tokens = analyzer._tokenize(test_prompt)
        
        # Generate and display visualization
        viz_output = analyzer.visualize_attention_flow(tokens, prediction)
        print(viz_output)
        
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    print("ATTENTION FLOW MODELING DEMONSTRATION")
    print("=====================================")
    
    # Run all demos
    demo_basic_attention_analysis()
    demo_context_window_analysis()
    demo_attention_competition_detection()
    demo_visualization()
    
    print("\n🎉 Attention Flow Modeling Demo Complete!")
    print("\nThis system can help you:")
    print("• Predict where LLMs will focus attention")
    print("• Identify conflicting instructions")  
    print("• Optimize prompt structure")
    print("• Improve context window utilization")
    print("• Detect attention bottlenecks")
    print("\nNext steps: Integrate with your LLM applications for better prompt engineering!")
