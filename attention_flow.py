"""Attention Flow Modeling for Static Prompt Analysis.

This module implements attention flow modeling to predict how transformers will
focus their attention on different parts of a prompt without running the actual model.
It uses linguistic features, token properties, and semantic relationships to simulate
attention patterns and detect potential conflicts or inefficiencies.
"""

from __future__ import annotations

import re
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class AttentionPrediction:
    """Predicted attention pattern for a token sequence."""
    token_importance: np.ndarray  # Importance score for each token
    attention_matrix: np.ndarray  # Predicted attention weights between tokens
    flow_values: np.ndarray      # Maximum flow values to each token
    competition_score: float     # Overall attention competition metric
    critical_tokens: List[int]   # Token indices with highest predicted attention
    attention_bottlenecks: List[Tuple[int, int]]  # Token pairs with attention conflicts


class AttentionFlowAnalyzer:
    """Analyzes and predicts attention flow patterns in prompts."""
    
    def __init__(self, embedder=None):
        """Initialize the attention flow analyzer.
        
        Parameters
        ----------
        embedder : OpenAIEmbeddings, optional
            Embedding model for semantic similarity computation.
        """
        self.embedder = embedder
        
        # Linguistic patterns that typically receive high attention
        self.high_attention_patterns = {
            'instructions': [r'\b(you must|you should|please|always|never)\b', r'\b(do not|don\'t)\b'],
            'questions': [r'\?', r'\b(what|how|when|where|why|which)\b'],
            'constraints': [r'\b(only|exactly|precisely|specifically)\b', r'\b(except|unless|but)\b'],
            'examples': [r'\b(for example|such as|like|including)\b', r':\s*[\"\']'],
            'emphasis': [r'\*\*.*?\*\*', r'\*.*?\*', r'[A-Z]{2,}'],
            'structure': [r'<[^>]+>', r'\n\n', r'^[0-9]+\.|^[\*\-]\s+'],
        }
        
        # Token types that compete for attention
        self.competition_sources = {
            'conflicting_instructions': [r'\bbut\b', r'\bhowever\b', r'\balthough\b'],
            'multiple_tasks': [r'\band\b.*\band\b', r'\bor\b.*\bor\b'],
            'nested_conditions': [r'\bif\b.*\bthen\b.*\belse\b'],
            'negations': [r'\bnot\b', r'\bno\b', r'\bnever\b'],
        }

    def analyze_attention_flow(self, text: str, tokens: Optional[List[str]] = None) -> AttentionPrediction:
        """Analyze predicted attention flow for a given text.
        
        Parameters
        ----------
        text : str
            Input text to analyze.
        tokens : List[str], optional
            Pre-tokenized input. If None, will tokenize the text.
            
        Returns
        -------
        AttentionPrediction
            Comprehensive attention flow analysis.
        """
        if tokens is None:
            tokens = self._tokenize(text)
            
        # Compute various attention factors
        positional_scores = self._compute_positional_importance(tokens)
        linguistic_scores = self._compute_linguistic_importance(tokens, text)
        semantic_scores = self._compute_semantic_importance(tokens, text)
        
        # Combine scores to predict token importance
        token_importance = self._combine_importance_scores(
            positional_scores, linguistic_scores, semantic_scores
        )
        
        # Predict attention matrix (token-to-token attention)
        attention_matrix = self._predict_attention_matrix(tokens, text, token_importance)
        
        # Compute attention flow using network flow algorithms
        flow_values = self._compute_attention_flow(attention_matrix)
        
        # Detect attention competition and conflicts
        competition_score = self._compute_attention_competition(tokens, text, attention_matrix)
        
        # Identify critical tokens and bottlenecks
        critical_tokens = self._identify_critical_tokens(token_importance, top_k=5)
        attention_bottlenecks = self._detect_attention_bottlenecks(attention_matrix, threshold=0.1)
        
        return AttentionPrediction(
            token_importance=token_importance,
            attention_matrix=attention_matrix,
            flow_values=flow_values,
            competition_score=competition_score,
            critical_tokens=critical_tokens,
            attention_bottlenecks=attention_bottlenecks
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for analysis."""
        # Split on whitespace and punctuation, keep punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    def _compute_positional_importance(self, tokens: List[str]) -> np.ndarray:
        """Compute importance based on token position.
        
        Transformers typically attend more to:
        - Beginning of sequence (instructions)
        - End of sequence (recent context)
        - Positions after punctuation (sentence starts)
        """
        n = len(tokens)
        scores = np.zeros(n)
        
        # Higher attention at beginning and end
        for i in range(n):
            # Exponential decay from start
            start_weight = np.exp(-i / (n * 0.3))
            # Exponential decay from end  
            end_weight = np.exp(-(n - 1 - i) / (n * 0.3))
            scores[i] = max(start_weight, end_weight * 0.5)
            
        # Boost after sentence boundaries
        for i in range(1, n):
            if tokens[i-1] in ['.', '!', '?', '\n']:
                scores[i] *= 1.5
                
        return scores / np.max(scores) if np.max(scores) > 0 else scores

    def _compute_linguistic_importance(self, tokens: List[str], text: str) -> np.ndarray:
        """Compute importance based on linguistic patterns."""
        n = len(tokens)
        scores = np.zeros(n)
        
        # Score based on high-attention patterns
        for pattern_type, patterns in self.high_attention_patterns.items():
            base_score = {
                'instructions': 1.0,
                'questions': 0.9, 
                'constraints': 0.8,
                'examples': 0.6,
                'emphasis': 0.7,
                'structure': 0.5,
            }[pattern_type]
            
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start_pos = len(text[:match.start()].split())
                    end_pos = len(text[:match.end()].split())
                    for i in range(max(0, start_pos), min(n, end_pos + 1)):
                        scores[i] += base_score
        
        # Special token types
        for i, token in enumerate(tokens):
            # Uppercase tokens (emphasis)
            if token.isupper() and len(token) > 1:
                scores[i] += 0.5
            # Numbers (specific values)
            elif token.isdigit():
                scores[i] += 0.3
            # Quoted content
            elif token.startswith('"') or token.startswith("'"):
                scores[i] += 0.4
                
        return scores

    def _compute_semantic_importance(self, tokens: List[str], text: str) -> np.ndarray:
        """Compute importance based on semantic content."""
        n = len(tokens)
        scores = np.zeros(n)
        
        if self.embedder is None:
            # Fallback: simple content word detection
            content_words = {'noun', 'verb', 'adjective', 'adverb'}  # Would use POS tagging in real implementation
            for i, token in enumerate(tokens):
                if len(token) > 3 and token.isalpha():
                    scores[i] = 0.5
        else:
            # Use embeddings to find semantically important tokens
            try:
                # Get embeddings for each sentence
                sentences = text.split('.')
                if len(sentences) > 1:
                    sentence_embeddings = [self.embedder.embed_query(sent.strip()) 
                                         for sent in sentences if sent.strip()]
                    # Compute average embedding
                    avg_embedding = np.mean(sentence_embeddings, axis=0)
                    
                    # Score tokens by their sentence's similarity to average
                    token_pos = 0
                    for sent in sentences:
                        if not sent.strip():
                            continue
                        sent_tokens = self._tokenize(sent)
                        sent_embedding = self.embedder.embed_query(sent.strip())
                        similarity = cosine_similarity([sent_embedding], [avg_embedding])[0][0]
                        
                        for _ in sent_tokens:
                            if token_pos < n:
                                scores[token_pos] += similarity
                                token_pos += 1
            except Exception:
                # Fallback on error
                pass
                
        return scores

    def _combine_importance_scores(self, positional: np.ndarray, linguistic: np.ndarray, 
                                 semantic: np.ndarray) -> np.ndarray:
        """Combine different importance scores into final token importance."""
        # Normalize each component
        pos_norm = positional / (np.max(positional) + 1e-8)
        ling_norm = linguistic / (np.max(linguistic) + 1e-8) 
        sem_norm = semantic / (np.max(semantic) + 1e-8)
        
        # Weighted combination (adjust weights based on your needs)
        combined = 0.3 * pos_norm + 0.5 * ling_norm + 0.2 * sem_norm
        
        # Apply softmax for final normalization
        return softmax(combined * 2)  # Temperature = 0.5

    def _predict_attention_matrix(self, tokens: List[str], text: str, 
                                token_importance: np.ndarray) -> np.ndarray:
        """Predict attention matrix between all token pairs."""
        n = len(tokens)
        attention = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Self-attention based on token importance
                    attention[i, j] = token_importance[i] * 0.5
                else:
                    # Cross-attention based on distance and importance
                    distance_penalty = 1.0 / (1.0 + abs(i - j) * 0.1)
                    semantic_boost = 1.0
                    
                    # Boost attention between related tokens
                    if self._tokens_are_related(tokens[i], tokens[j]):
                        semantic_boost = 1.5
                        
                    attention[i, j] = (token_importance[j] * distance_penalty * semantic_boost) * 0.1
        
        # Normalize each row to sum to 1
        for i in range(n):
            row_sum = np.sum(attention[i, :])
            if row_sum > 0:
                attention[i, :] /= row_sum
                
        return attention

    def _tokens_are_related(self, token1: str, token2: str) -> bool:
        """Simple heuristic to determine if two tokens are semantically related."""
        # Same word family
        if token1.lower().startswith(token2.lower()[:3]) or token2.lower().startswith(token1.lower()[:3]):
            return True
        # Both are content words (simple heuristic)
        if len(token1) > 3 and len(token2) > 3 and token1.isalpha() and token2.isalpha():
            return True
        return False

    def _compute_attention_flow(self, attention_matrix: np.ndarray) -> np.ndarray:
        """Compute maximum flow values using the attention matrix as capacities."""
        n = attention_matrix.shape[0]
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(n):
            G.add_node(i)
        
        # Add edges with capacities based on attention weights
        for i in range(n):
            for j in range(n):
                if attention_matrix[i, j] > 0.01:  # Threshold for including edges
                    G.add_edge(i, j, capacity=attention_matrix[i, j])
        
        # Compute flow values (sum of flows from each node to all others)
        flow_values = np.zeros(n)
        for source in range(n):
            total_flow = 0
            for target in range(n):
                if source != target and G.has_node(source) and G.has_node(target):
                    try:
                        flow_value = nx.maximum_flow_value(G, source, target)
                        total_flow += flow_value
                    except nx.NetworkXNoPath:
                        continue
            flow_values[source] = total_flow
            
        return flow_values

    def _compute_attention_competition(self, tokens: List[str], text: str, 
                                     attention_matrix: np.ndarray) -> float:
        """Compute overall attention competition score."""
        competition = 0.0
        
        # Look for competing patterns
        for pattern_type, patterns in self.competition_sources.items():
            matches = []
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    matches.append((match.start(), match.end()))
            
            if len(matches) > 1:
                # Multiple competing elements found
                weight = {
                    'conflicting_instructions': 1.0,
                    'multiple_tasks': 0.8,
                    'nested_conditions': 0.6,
                    'negations': 0.4,
                }[pattern_type]
                competition += len(matches) * weight
        
        # Analyze attention dispersion
        attention_entropy = 0.0
        for i in range(attention_matrix.shape[0]):
            row = attention_matrix[i, :]
            row_entropy = -np.sum(row * np.log(row + 1e-8))
            attention_entropy += row_entropy
        
        # High entropy = more competition
        normalized_entropy = attention_entropy / attention_matrix.shape[0]
        competition += normalized_entropy * 0.5
        
        return min(competition, 10.0)  # Cap at reasonable maximum

    def _identify_critical_tokens(self, token_importance: np.ndarray, top_k: int = 5) -> List[int]:
        """Identify the most critical tokens based on importance scores."""
        return list(np.argsort(token_importance)[-top_k:][::-1])

    def _detect_attention_bottlenecks(self, attention_matrix: np.ndarray, 
                                    threshold: float = 0.1) -> List[Tuple[int, int]]:
        """Detect token pairs that create attention bottlenecks."""
        bottlenecks = []
        n = attention_matrix.shape[0]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Look for asymmetric high attention (potential conflict)
                    att_ij = attention_matrix[i, j]
                    att_ji = attention_matrix[j, i]
                    
                    if att_ij > threshold and att_ji > threshold and abs(att_ij - att_ji) > threshold:
                        bottlenecks.append((i, j))
        
        return bottlenecks

    def visualize_attention_flow(self, tokens: List[str], prediction: AttentionPrediction) -> str:
        """Generate a text-based visualization of attention flow."""
        output = []
        output.append("=== ATTENTION FLOW ANALYSIS ===\n")
        
        # Token importance
        output.append("TOKEN IMPORTANCE SCORES:")
        for i, (token, score) in enumerate(zip(tokens, prediction.token_importance)):
            bar = "█" * int(score * 20)  # Visual bar
            output.append(f"{i:3d}: {token:15s} {score:.3f} {bar}")
        
        output.append(f"\nCOMPETITION SCORE: {prediction.competition_score:.3f}")
        
        # Critical tokens
        output.append(f"\nCRITICAL TOKENS:")
        for idx in prediction.critical_tokens:
            output.append(f"  {idx}: {tokens[idx]} (importance: {prediction.token_importance[idx]:.3f})")
        
        # Attention bottlenecks
        if prediction.attention_bottlenecks:
            output.append(f"\nATTENTION BOTTLENECKS:")
            for i, j in prediction.attention_bottlenecks[:5]:  # Show top 5
                output.append(f"  {tokens[i]} <-> {tokens[j]}")
        
        return "\n".join(output)
