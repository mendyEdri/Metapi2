"""
Semantic similarity processing for system prompt analysis.
Focuses on meaning rather than syntactic structure.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Set
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSimilarityAnalyzer:
    """Analyzes semantic similarity between text chunks, focusing on meaning over syntax."""
    
    def __init__(self):
        # Define syntactic words that dilute semantic meaning
        self.syntax_words = {
            'you', 'might', 'can', 'should', 'must', 'will', 'would', 'could',
            'be', 'is', 'are', 'am', 'was', 'were', 'have', 'has', 'had',
            'and', 'or', 'but', 'the', 'a', 'an', 'this', 'that', 'these', 'those',
            'from', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'about',
            'time', 'some', 'any', 'all', 'each', 'every', 'many', 'much',
            'answer', 'respond', 'reply', 'speak', 'speaks', 'speaking', 'say', 'tell'
        }
        
        # Define system prompt concept categories
        self.concept_patterns = {
            'languages': {
                'patterns': [r'\b(spanish|english|hebrew|french|german|chinese|japanese|korean|arabic|hindi|russian|portuguese|italian)\b'],
                'weight': 2.0
            },
            'behaviors': {
                'patterns': [r'\b(helpful|nice|professional|creative|respectful|courteous|friendly|polite|kind|caring)\b'],
                'weight': 1.5
            },
            'actions': {
                'patterns': [r'\b(translate|write|create|generate|explain|analyze|summarize|describe)\b'],
                'weight': 1.2
            },
            'constraints': {
                'patterns': [r'\b(never|always|only|exclusively|strictly|must|required|forbidden|prohibited)\b'],
                'weight': 2.5
            },
            'negations': {
                'patterns': [r'\b(never ever|never|not|don\'t|cannot|can\'t|shouldn\'t|won\'t)\b'],
                'weight': 2.0
            },
            'topics': {
                'patterns': [r'\b(information|knowledge|data|facts|details|content|subject|topic)\b'],
                'weight': 1.0
            }
        }
        
    def extract_semantic_keywords(self, text: str) -> str:
        """Extract meaningful keywords, removing syntactic noise."""
        # Convert to lowercase for processing
        text_lower = text.lower()
        
        # Extract all words
        words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
        
        # Filter out syntax words and short words
        meaningful_words = [
            word for word in words 
            if word not in self.syntax_words and len(word) > 2
        ]
        
        # If no meaningful words found, return original text
        if not meaningful_words:
            return text
            
        return ' '.join(meaningful_words)
    
    def extract_system_prompt_concepts(self, text: str) -> Dict[str, List[str]]:
        """Extract domain-specific concepts from system prompts."""
        text_lower = text.lower()
        extracted_concepts = {}
        
        for concept_type, config in self.concept_patterns.items():
            matches = []
            for pattern in config['patterns']:
                found_matches = re.findall(pattern, text_lower)
                matches.extend(found_matches)
            extracted_concepts[concept_type] = matches
            
        return extracted_concepts
    
    def create_concept_vector(self, text: str) -> str:
        """Create a weighted concept representation for embedding."""
        concepts = self.extract_system_prompt_concepts(text)
        
        concept_terms = []
        for concept_type, matches in concepts.items():
            if matches:
                weight = self.concept_patterns[concept_type]['weight']
                # Repeat terms based on weight to increase their embedding influence
                weighted_terms = []
                for match in matches:
                    repetitions = max(1, int(weight))
                    weighted_terms.extend([f"{concept_type}_{match}"] * repetitions)
                concept_terms.extend(weighted_terms)
        
        # Add semantic keywords as baseline
        semantic_keywords = self.extract_semantic_keywords(text)
        if semantic_keywords:
            concept_terms.extend(semantic_keywords.split())
            
        return ' '.join(concept_terms) if concept_terms else text
    
    def compute_semantic_similarity(self, chunks: List[str], embedder, method: str = 'concepts') -> np.ndarray:
        """
        Compute semantic similarity matrix using different methods.
        
        Args:
            chunks: List of text chunks
            embedder: OpenAI embeddings instance
            method: 'concepts' (domain-aware), 'keywords' (general semantic), or 'combined'
        
        Returns:
            Similarity matrix based on semantic content
        """
        if method == 'concepts':
            # Use system prompt concept extraction
            processed_chunks = [self.create_concept_vector(chunk) for chunk in chunks]
        elif method == 'keywords':
            # Use general semantic keyword extraction
            processed_chunks = [self.extract_semantic_keywords(chunk) for chunk in chunks]
        elif method == 'combined':
            # Combine both approaches
            concept_chunks = [self.create_concept_vector(chunk) for chunk in chunks]
            keyword_chunks = [self.extract_semantic_keywords(chunk) for chunk in chunks]
            processed_chunks = [f"{concept} {keyword}" for concept, keyword in zip(concept_chunks, keyword_chunks)]
        else:
            processed_chunks = chunks  # Fallback to original
            
        # Get embeddings for processed chunks
        embeddings = []
        for processed_chunk in processed_chunks:
            # Ensure we have meaningful content to embed
            if not processed_chunk.strip():
                processed_chunk = chunks[len(embeddings)]  # Use original if processing failed
            embedding = embedder.embed_query(processed_chunk)
            embeddings.append(embedding)
        
        # Compute similarity matrix
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        return similarity_matrix, processed_chunks
    
    def compare_similarity_methods(self, chunks: List[str], embedder) -> Dict[str, Tuple[np.ndarray, List[str]]]:
        """Compare different similarity computation methods."""
        results = {}
        
        # Original syntactic similarity
        original_embeddings = []
        for chunk in chunks:
            embedding = embedder.embed_query(chunk)
            original_embeddings.append(embedding)
        original_similarity = cosine_similarity(np.array(original_embeddings))
        results['syntactic'] = (original_similarity, chunks)
        
        # Semantic methods
        for method in ['keywords', 'concepts', 'combined']:
            similarity_matrix, processed_chunks = self.compute_semantic_similarity(chunks, embedder, method)
            results[method] = (similarity_matrix, processed_chunks)
            
        return results
    
    def analyze_similarity_differences(self, chunks: List[str], syntactic_matrix: np.ndarray, semantic_matrix: np.ndarray) -> Dict:
        """Analyze the differences between syntactic and semantic similarity."""
        n_chunks = len(chunks)
        
        # Find pairs where semantic and syntactic similarity differ significantly
        significant_differences = []
        threshold = 0.2  # Minimum difference to consider significant
        
        for i in range(n_chunks):
            for j in range(i + 1, n_chunks):
                syntactic_sim = syntactic_matrix[i, j]
                semantic_sim = semantic_matrix[i, j]
                difference = abs(syntactic_sim - semantic_sim)
                
                if difference > threshold:
                    significant_differences.append({
                        'chunk_pair': (i, j),
                        'syntactic_similarity': syntactic_sim,
                        'semantic_similarity': semantic_sim,
                        'difference': difference,
                        'type': 'semantic_higher' if semantic_sim > syntactic_sim else 'syntactic_higher'
                    })
        
        # Sort by difference magnitude
        significant_differences.sort(key=lambda x: x['difference'], reverse=True)
        
        # Find most similar pairs for each method
        syntactic_max = np.unravel_index(
            np.argmax(np.triu(syntactic_matrix, k=1)), syntactic_matrix.shape
        )
        semantic_max = np.unravel_index(
            np.argmax(np.triu(semantic_matrix, k=1)), semantic_matrix.shape
        )
        
        return {
            'significant_differences': significant_differences[:5],  # Top 5
            'syntactic_most_similar': {
                'pair': syntactic_max,
                'similarity': syntactic_matrix[syntactic_max]
            },
            'semantic_most_similar': {
                'pair': semantic_max,
                'similarity': semantic_matrix[semantic_max]
            }
        }
    
    def generate_similarity_insights(self, chunks: List[str], analysis_results: Dict) -> List[str]:
        """Generate human-readable insights about similarity differences."""
        insights = []
        
        syntactic_pair = analysis_results['syntactic_most_similar']['pair']
        semantic_pair = analysis_results['semantic_most_similar']['pair']
        
        if syntactic_pair != semantic_pair:
            insights.append(
                f"🔄 **Method Disagreement**: Syntactic analysis finds Chunks {syntactic_pair[0]+1} & {syntactic_pair[1]+1} most similar, "
                f"but semantic analysis finds Chunks {semantic_pair[0]+1} & {semantic_pair[1]+1} most similar."
            )
            
            # Show what each method is detecting
            chunk1, chunk2 = chunks[syntactic_pair[0]], chunks[syntactic_pair[1]]
            insights.append(f"   - **Syntactic similarity**: '{chunk1[:30]}...' ↔ '{chunk2[:30]}...' (sentence structure)")
            
            chunk1, chunk2 = chunks[semantic_pair[0]], chunks[semantic_pair[1]]  
            insights.append(f"   - **Semantic similarity**: '{chunk1[:30]}...' ↔ '{chunk2[:30]}...' (meaning/concepts)")
        
        # Analyze significant differences
        if analysis_results['significant_differences']:
            insights.append("🎯 **Key Differences Found:**")
            for diff in analysis_results['significant_differences'][:3]:
                i, j = diff['chunk_pair']
                if diff['type'] == 'semantic_higher':
                    insights.append(
                        f"   - Chunks {i+1} & {j+1}: Semantically similar ({diff['semantic_similarity']:.3f}) "
                        f"but syntactically different ({diff['syntactic_similarity']:.3f})"
                    )
                else:
                    insights.append(
                        f"   - Chunks {i+1} & {j+1}: Syntactically similar ({diff['syntactic_similarity']:.3f}) "
                        f"but semantically different ({diff['semantic_similarity']:.3f})"
                    )
        
        if not insights:
            insights.append("✅ **Consistent Results**: Both methods show similar patterns - good semantic-syntactic alignment.")
            
        return insights
