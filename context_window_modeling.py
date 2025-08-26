"""Context Window Modeling for Attention Flow Analysis.

This module extends attention flow modeling to account for context window
effects, including attention decay over distance, window utilization patterns,
and the impact of prompt structure on information processing.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ContextWindowAnalysis:
    """Analysis of how prompt structure affects context window utilization."""
    effective_length: int           # Effective context length after attention decay
    utilization_score: float       # How well the context window is utilized (0-1)
    attention_decay_curve: np.ndarray  # Attention decay across positions
    information_density: np.ndarray    # Information density per position
    bottleneck_positions: List[int]    # Positions where attention bottlenecks occur
    optimal_structure_score: float    # How well-structured the prompt is (0-1)


class ContextWindowModeler:
    """Models how prompt length and structure affects attention distribution."""
    
    def __init__(self, max_context_length: int = 4096):
        """Initialize context window modeler.
        
        Parameters
        ----------
        max_context_length : int
            Maximum context length to model (in tokens).
        """
        self.max_context_length = max_context_length
        
        # Empirical parameters based on transformer research
        self.attention_decay_rate = 0.95  # Attention decay per position
        self.recency_bias = 0.1          # Boost for recent tokens
        self.structure_bonus = 0.2       # Bonus for well-structured content
        
    def analyze_context_window_usage(self, tokens: List[str], 
                                   attention_matrix: np.ndarray) -> ContextWindowAnalysis:
        """Analyze how effectively the context window is being used.
        
        Parameters
        ----------
        tokens : List[str]
            Input tokens
        attention_matrix : np.ndarray
            Predicted attention matrix from AttentionFlowAnalyzer
            
        Returns
        -------
        ContextWindowAnalysis
            Comprehensive analysis of context window utilization
        """
        n_tokens = len(tokens)
        
        # Compute attention decay curve
        decay_curve = self._compute_attention_decay(n_tokens)
        
        # Analyze information density
        info_density = self._compute_information_density(tokens)
        
        # Calculate effective context length
        effective_length = self._calculate_effective_length(decay_curve, info_density)
        
        # Compute utilization score
        utilization_score = self._compute_utilization_score(
            n_tokens, effective_length, attention_matrix
        )
        
        # Find attention bottlenecks
        bottlenecks = self._find_bottleneck_positions(attention_matrix, decay_curve)
        
        # Score prompt structure optimality
        structure_score = self._score_prompt_structure(tokens, attention_matrix)
        
        return ContextWindowAnalysis(
            effective_length=effective_length,
            utilization_score=utilization_score,
            attention_decay_curve=decay_curve,
            information_density=info_density,
            bottleneck_positions=bottlenecks,
            optimal_structure_score=structure_score
        )
    
    def _compute_attention_decay(self, n_tokens: int) -> np.ndarray:
        """Compute how attention decays across token positions."""
        positions = np.arange(n_tokens)
        
        # Exponential decay from start
        forward_decay = np.power(self.attention_decay_rate, positions)
        
        # Recency bias (boost for recent tokens)
        recency_positions = n_tokens - 1 - positions
        recency_boost = 1.0 + self.recency_bias * np.exp(-recency_positions / (n_tokens * 0.1))
        
        # Combine forward decay and recency bias
        attention_capacity = forward_decay * recency_boost
        
        # Normalize
        return attention_capacity / np.max(attention_capacity)
    
    def _compute_information_density(self, tokens: List[str]) -> np.ndarray:
        """Compute information density at each token position."""
        n_tokens = len(tokens)
        density = np.ones(n_tokens)
        
        # Content words have higher density
        for i, token in enumerate(tokens):
            if len(token) > 3 and token.isalpha():
                density[i] *= 1.5
            elif token.isdigit():
                density[i] *= 1.3
            elif token in ['.', '!', '?']:
                density[i] *= 0.8  # Punctuation has lower density
            elif token in ['the', 'a', 'an', 'and', 'or', 'but']:
                density[i] *= 0.7  # Function words have lower density
        
        # Boost density after structure markers
        for i in range(1, n_tokens):
            prev_token = tokens[i-1]
            if prev_token in ['.', ':', '\n'] or prev_token.isdigit():
                density[i] *= 1.2  # Information often follows structure
        
        return density / np.max(density)
    
    def _calculate_effective_length(self, decay_curve: np.ndarray, 
                                  info_density: np.ndarray) -> int:
        """Calculate the effective context length considering attention decay."""
        # Combine attention capacity with information density
        effective_attention = decay_curve * info_density
        
        # Find where effective attention drops below threshold
        threshold = 0.1 * np.max(effective_attention)
        effective_positions = np.where(effective_attention >= threshold)[0]
        
        if len(effective_positions) == 0:
            return 1
        
        return int(effective_positions[-1]) + 1
    
    def _compute_utilization_score(self, n_tokens: int, effective_length: int, 
                                 attention_matrix: np.ndarray) -> float:
        """Compute how well the available context window is utilized."""
        if n_tokens == 0:
            return 0.0
            
        # Base utilization: how much of available context is effectively used
        length_utilization = min(effective_length / n_tokens, 1.0)
        
        # Attention distribution efficiency
        attention_entropy = 0.0
        for i in range(attention_matrix.shape[0]):
            row = attention_matrix[i, :]
            # Skip zero entries for entropy calculation
            nonzero_probs = row[row > 1e-8]
            if len(nonzero_probs) > 0:
                attention_entropy -= np.sum(nonzero_probs * np.log(nonzero_probs))
        
        # Normalize entropy by maximum possible entropy
        max_entropy = np.log(n_tokens) if n_tokens > 1 else 1.0
        attention_efficiency = 1.0 - (attention_entropy / (n_tokens * max_entropy))
        
        # Combine metrics
        utilization_score = 0.6 * length_utilization + 0.4 * attention_efficiency
        
        return np.clip(utilization_score, 0.0, 1.0)
    
    def _find_bottleneck_positions(self, attention_matrix: np.ndarray, 
                                 decay_curve: np.ndarray) -> List[int]:
        """Find positions where attention bottlenecks occur."""
        bottlenecks = []
        n_tokens = attention_matrix.shape[0]
        
        for i in range(n_tokens):
            # Calculate attention flow through this position
            incoming_attention = np.sum(attention_matrix[:, i])
            outgoing_attention = np.sum(attention_matrix[i, :])
            expected_attention = decay_curve[i]
            
            # Bottleneck if attention flow is much lower than expected
            if incoming_attention < 0.5 * expected_attention or outgoing_attention < 0.5 * expected_attention:
                bottlenecks.append(i)
        
        return bottlenecks
    
    def _score_prompt_structure(self, tokens: List[str], 
                              attention_matrix: np.ndarray) -> float:
        """Score how well-structured the prompt is for optimal attention flow."""
        n_tokens = len(tokens)
        if n_tokens == 0:
            return 0.0
        
        structure_score = 0.0
        
        # 1. Check for clear section breaks
        section_breaks = 0
        for i, token in enumerate(tokens):
            if token in ['.', ':', '\n'] or (token.isdigit() and i > 0):
                section_breaks += 1
        
        # Optimal number of sections for readability
        optimal_sections = max(2, n_tokens // 20)
        section_score = 1.0 - abs(section_breaks - optimal_sections) / optimal_sections
        structure_score += 0.3 * section_score
        
        # 2. Check instruction clarity (imperatives at beginning)
        instruction_words = ['you', 'must', 'should', 'always', 'never', 'please']
        early_instructions = 0
        for i, token in enumerate(tokens[:min(10, n_tokens)]):
            if token.lower() in instruction_words:
                early_instructions += 1
        
        instruction_score = min(early_instructions / 3.0, 1.0)  # Normalize to 0-1
        structure_score += 0.25 * instruction_score
        
        # 3. Check for examples placement (should be after instructions)
        example_indicators = ['example', 'like', 'such as', 'for instance']
        examples_after_instructions = 0
        instructions_found = False
        
        for i, token in enumerate(tokens):
            if token.lower() in instruction_words:
                instructions_found = True
            elif instructions_found and token.lower() in example_indicators:
                examples_after_instructions += 1
        
        example_score = min(examples_after_instructions / 2.0, 1.0)
        structure_score += 0.2 * example_score
        
        # 4. Attention flow smoothness
        attention_variance = 0.0
        for i in range(n_tokens - 1):
            attention_diff = abs(np.sum(attention_matrix[i, :]) - np.sum(attention_matrix[i+1, :]))
            attention_variance += attention_diff
        
        if n_tokens > 1:
            attention_variance /= (n_tokens - 1)
        
        # Lower variance = smoother attention flow = better structure
        smoothness_score = 1.0 / (1.0 + attention_variance * 10)
        structure_score += 0.25 * smoothness_score
        
        return np.clip(structure_score, 0.0, 1.0)
    
    def generate_optimization_suggestions(self, analysis: ContextWindowAnalysis, 
                                        tokens: List[str]) -> List[str]:
        """Generate suggestions for improving context window utilization."""
        suggestions = []
        
        # Check utilization score
        if analysis.utilization_score < 0.6:
            suggestions.append(
                f"Low context utilization ({analysis.utilization_score:.2f}). "
                "Consider restructuring to make better use of available context length."
            )
        
        # Check effective length
        token_count = len(tokens)
        if analysis.effective_length < 0.7 * token_count:
            suggestions.append(
                f"Only {analysis.effective_length}/{token_count} tokens are effectively processed. "
                "Move important information earlier in the prompt."
            )
        
        # Check structure score
        if analysis.optimal_structure_score < 0.7:
            suggestions.append(
                f"Suboptimal prompt structure (score: {analysis.optimal_structure_score:.2f}). "
                "Consider adding clear section breaks and placing instructions at the beginning."
            )
        
        # Check bottlenecks
        if len(analysis.bottleneck_positions) > 3:
            suggestions.append(
                f"Found {len(analysis.bottleneck_positions)} attention bottlenecks. "
                "Simplify complex sections or break them into smaller parts."
            )
        
        # Check information density distribution
        if np.std(analysis.information_density) > 0.3:
            suggestions.append(
                "Uneven information distribution detected. "
                "Balance content density across the prompt for better attention flow."
            )
        
        if not suggestions:
            suggestions.append("Prompt structure looks well-optimized for attention flow!")
        
        return suggestions
