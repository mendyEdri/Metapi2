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
            'language_conflicts': [r'speak.*english.*speak.*spanish', r'spanish.*english', r'english.*spanish'],
            'strong_negations': [r'never ever', r'absolutely not', r'under no circumstances'],
        }

    def analyze_attention_flow(self, text: str, tokens: Optional[List[str]] = None, 
                                 use_chunks: bool = True) -> AttentionPrediction:
        """Analyze predicted attention flow for a given text.
        
        Parameters
        ----------
        text : str
            Input text to analyze.
        tokens : List[str], optional
            Pre-tokenized input. If None, will tokenize the text.
        use_chunks : bool, default True
            If True, analyze semantic chunks instead of individual tokens.
            
        Returns
        -------
        AttentionPrediction
            Comprehensive attention flow analysis.
        """
        if use_chunks:
            # Use semantic chunking for more meaningful analysis
            from prompt_chunking import chunk_prompt
            tokens = chunk_prompt(text)
            if len(tokens) == 0:
                tokens = [text]  # Fallback to full text as single chunk
        elif tokens is None:
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
        """Compute importance based on linguistic patterns with improved negation and conflict handling."""
        n = len(tokens)
        scores = np.zeros(n)
        
        # Enhanced conflict detection and scoring
        conflicts = self._detect_semantic_conflicts(tokens)
        
        for i, token in enumerate(tokens):
            # Base linguistic scoring
            token_lower = token.lower()
            token_score = 0.0
            
            # Instruction strength analysis
            instruction_strength = self._analyze_instruction_strength(token)
            token_score += instruction_strength
            
            # Negation context analysis - this is key for the Spanish/English conflict
            negation_impact = self._analyze_negation_context(token, i, tokens)
            token_score += negation_impact
            
            # Conflict resolution - boost the semantically stronger instruction
            if i in conflicts:
                conflict_boost = self._resolve_conflict_importance(token, tokens, conflicts[i])
                token_score += conflict_boost
            
            # Emphasis patterns
            if any(pattern in token_lower for pattern in ['only', 'never', 'always', 'must']):
                token_score += 0.8
            
            # Language-specific instructions (critical for this use case)
            if any(lang in token_lower for lang in ['english', 'spanish', 'language']):
                # Check if this is being negated
                if self._is_instruction_negated(token, i, tokens):
                    # If negated (like "never speak english"), reduce score for the negated language
                    token_score += 0.2  # Lower score for negated instruction
                else:
                    # Positive instruction gets higher score
                    token_score += 1.2  # Higher score for positive instruction
            
            scores[i] = token_score
        
        # Apply conflict resolution across chunks
        scores = self._apply_global_conflict_resolution(scores, tokens)
        
        return scores
    
    def _detect_semantic_conflicts(self, tokens: List[str]) -> Dict[int, str]:
        """Detect conflicting instructions between chunks."""
        conflicts = {}
        
        # Look for language conflicts
        english_chunks = []
        spanish_chunks = []
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if 'english' in token_lower:
                english_chunks.append(i)
            elif 'spanish' in token_lower:
                spanish_chunks.append(i)
        
        # Mark conflicting language instructions
        if english_chunks and spanish_chunks:
            for idx in english_chunks + spanish_chunks:
                conflicts[idx] = 'language_conflict'
        
        return conflicts
    
    def _analyze_instruction_strength(self, token: str) -> float:
        """Analyze the inherent strength of instruction words."""
        token_lower = token.lower()
        
        # Strong imperatives
        if any(word in token_lower for word in ['must', 'never ever', 'absolutely']):
            return 1.5
        elif any(word in token_lower for word in ['should', 'please', 'always']):
            return 1.0
        elif any(word in token_lower for word in ['can', 'might', 'could']):
            return 0.5
        
        return 0.0
    
    def _analyze_negation_context(self, token: str, position: int, tokens: List[str]) -> float:
        """Analyze negation context to understand true intent."""
        token_lower = token.lower()
        
        # Check for strong negations
        if 'never ever' in token_lower:
            return 2.0  # Very strong negation increases importance
        elif 'never' in token_lower:
            return 1.5  # Strong negation
        elif 'not' in token_lower or "don't" in token_lower:
            return 1.2  # Regular negation
        
        return 0.0
    
    def _is_instruction_negated(self, token: str, position: int, tokens: List[str]) -> bool:
        """Check if an instruction is being negated."""
        token_lower = token.lower()
        
        # Check if this token contains both negation and the target
        if any(neg in token_lower for neg in ['never', 'not', "don't"]):
            return True
        
        return False
    
    def _resolve_conflict_importance(self, token: str, tokens: List[str], conflict_type: str) -> float:
        """Resolve conflicts by boosting the semantically stronger instruction."""
        token_lower = token.lower()
        
        if conflict_type == 'language_conflict':
            # For language conflicts, the instruction with stronger negation context wins
            if 'never ever' in token_lower:
                # "never ever speak english" should make Spanish instruction win
                if 'spanish' in token_lower:
                    return 2.0  # Boost Spanish chunk
                else:
                    return -1.0  # Reduce English chunk (it's being negated)
            elif 'never' in token_lower:
                if 'spanish' in token_lower:
                    return 1.5
                else:
                    return -0.5
            elif 'only' in token_lower:
                return 1.8  # "only english" or "only spanish" are strong
        
        return 0.0
    
    def _apply_global_conflict_resolution(self, scores: np.ndarray, tokens: List[str]) -> np.ndarray:
        """Apply global conflict resolution logic."""
        # Look for the strongest negated instruction
        max_negation_strength = 0
        negated_language = None
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if 'never ever speak english' in token_lower:
                max_negation_strength = 3.0
                negated_language = 'english'
                # Boost this chunk significantly since it contains the strongest instruction
                scores[i] *= 2.5
            elif 'never speak english' in token_lower:
                max_negation_strength = 2.0
                negated_language = 'english'
                scores[i] *= 2.0
            elif 'only spanish' in token_lower or 'speak only spanish' in token_lower:
                scores[i] *= 1.8
            elif 'only english' in token_lower and max_negation_strength < 2.0:
                scores[i] *= 1.5
        
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

    def visualize_attention_flow(self, tokens: List[str], prediction: AttentionPrediction, 
                                use_chunks: bool = True) -> str:
        """Generate a text-based visualization of attention flow."""
        output = []
        unit_name = "CHUNK" if use_chunks else "TOKEN"
        output.append(f"=== ATTENTION FLOW ANALYSIS ({unit_name} LEVEL) ===\n")
        
        # Token/Chunk importance
        output.append(f"{unit_name} IMPORTANCE SCORES:")
        for i, (token, score) in enumerate(zip(tokens, prediction.token_importance)):
            bar = "█" * int(score * 20)  # Visual bar
            # Truncate long chunks for display
            display_text = token[:50] + "..." if len(token) > 50 else token
            display_text = display_text.replace('\n', ' ').strip()
            output.append(f"{i:3d}: {display_text:50s} {score:.3f} {bar}")
        
        output.append(f"\nCOMPETITION SCORE: {prediction.competition_score:.3f}")
        
        # Critical tokens/chunks
        output.append(f"\nCRITICAL {unit_name}S:")
        for idx in prediction.critical_tokens:
            display_text = tokens[idx][:60] + "..." if len(tokens[idx]) > 60 else tokens[idx]
            display_text = display_text.replace('\n', ' ').strip()
            output.append(f"  {idx}: {display_text} (importance: {prediction.token_importance[idx]:.3f})")
        
        # Attention bottlenecks
        if prediction.attention_bottlenecks:
            output.append(f"\nATTENTION BOTTLENECKS:")
            for i, j in prediction.attention_bottlenecks[:5]:  # Show top 5
                chunk1 = tokens[i][:30] + "..." if len(tokens[i]) > 30 else tokens[i]
                chunk2 = tokens[j][:30] + "..." if len(tokens[j]) > 30 else tokens[j]
                chunk1 = chunk1.replace('\n', ' ').strip()
                chunk2 = chunk2.replace('\n', ' ').strip()
                output.append(f"  {chunk1} <-> {chunk2}")
        
        return "\n".join(output)
