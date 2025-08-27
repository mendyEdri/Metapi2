"""
Semantic similarity processing for system prompt analysis.
Focuses on meaning rather than syntactic structure.
"""

import re
import numpy as np
import math  # For log2 in entropy calculation
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

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
    
    def extract_prompt_concepts(self, text: str) -> Dict[str, List[str]]:
        """Alias for extract_system_prompt_concepts for compatibility."""
        return self.extract_system_prompt_concepts(text)
    
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

    def perform_clustering_analysis(self, chunks: List[str], similarity_matrix: np.ndarray, 
                                  method: str = 'hierarchical', n_clusters: Optional[int] = None) -> Dict:
        """
        Perform clustering analysis on chunks to identify thematic groups.
        
        Args:
            chunks: List of text chunks
            similarity_matrix: Pairwise similarity matrix
            method: 'hierarchical', 'kmeans', or 'auto'
            n_clusters: Number of clusters (None for auto-detection)
        
        Returns:
            Dictionary with clustering results and analysis
        """
        # Convert similarity to distance matrix for clustering
        # Ensure similarity values are in [0,1] range to avoid negative distances
        similarity_matrix_clipped = np.clip(similarity_matrix, 0, 1)
        distance_matrix = 1 - similarity_matrix_clipped
        
        # Auto-detect optimal number of clusters if not specified
        if n_clusters is None or method == 'auto':
            n_clusters = self._find_optimal_clusters(distance_matrix, max_clusters=min(8, len(chunks)-1))
        
        # Perform clustering
        if method == 'hierarchical' or method == 'auto':
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
        else:  # kmeans
            # Use original similarity matrix for kmeans (convert to embeddings-like format)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
        if method == 'kmeans':
            # For kmeans, use the similarity matrix as pseudo-embeddings
            cluster_labels = clusterer.fit_predict(similarity_matrix)
        else:
            # For hierarchical, use distance matrix
            cluster_labels = clusterer.fit_predict(distance_matrix)
        
        # Calculate clustering quality metrics
        if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
            try:
                if method == 'kmeans':
                    silhouette = silhouette_score(similarity_matrix_clipped, cluster_labels)
                else:
                    silhouette = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
            except Exception as e:
                # Fallback if silhouette calculation fails
                silhouette = 0.0
        else:
            silhouette = 0.0
        
        # Organize results
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        
        # Generate cluster themes and insights
        cluster_analysis = self._analyze_clusters(chunks, clusters, similarity_matrix)
        
        return {
            'labels': cluster_labels.tolist(),
            'n_clusters': n_clusters,
            'clusters': clusters,
            'silhouette_score': silhouette,
            'method_used': method,
            'cluster_analysis': cluster_analysis,
            'distance_matrix': distance_matrix
        }
    
    def _find_optimal_clusters(self, distance_matrix: np.ndarray, max_clusters: int = 6) -> int:
        """Find optimal number of clusters using silhouette analysis."""
        if len(distance_matrix) < 3:
            return max(1, len(distance_matrix) - 1)
        
        best_score = -1
        best_k = 2
        
        for k in range(2, min(max_clusters + 1, len(distance_matrix))):
            clusterer = AgglomerativeClustering(
                n_clusters=k,
                metric='precomputed', 
                linkage='average'
            )
            cluster_labels = clusterer.fit_predict(distance_matrix)
            try:
                score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
            except Exception:
                # Skip this k if clustering fails
                continue
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k
    
    def _analyze_clusters(self, chunks: List[str], clusters: Dict, similarity_matrix: np.ndarray) -> Dict:
        """Analyze clusters to extract themes and generate insights."""
        analysis = {}
        
        for cluster_id, chunk_indices in clusters.items():
            cluster_chunks = [chunks[i] for i in chunk_indices]
            
            # Extract common themes/concepts for this cluster
            cluster_concepts = {}
            for chunk_idx in chunk_indices:
                chunk_concepts = self.extract_prompt_concepts(chunks[chunk_idx])
                for concept_type, items in chunk_concepts.items():
                    if concept_type not in cluster_concepts:
                        cluster_concepts[concept_type] = set()
                    cluster_concepts[concept_type].update(items)
            
            # Convert sets back to lists
            cluster_concepts = {k: list(v) for k, v in cluster_concepts.items() if v}
            
            # Calculate intra-cluster similarity (cohesion)
            if len(chunk_indices) > 1:
                similarities = []
                for i in range(len(chunk_indices)):
                    for j in range(i + 1, len(chunk_indices)):
                        similarities.append(similarity_matrix[chunk_indices[i], chunk_indices[j]])
                cohesion = np.mean(similarities) if similarities else 0
            else:
                cohesion = 1.0  # Single chunk cluster is perfectly cohesive
            
            # Generate cluster theme name
            theme = self._generate_cluster_theme(cluster_concepts, cluster_chunks)
            
            analysis[cluster_id] = {
                'theme': theme,
                'chunks': cluster_chunks,
                'chunk_indices': chunk_indices,
                'concepts': cluster_concepts,
                'cohesion': cohesion,
                'size': len(chunk_indices)
            }
        
        return analysis
    
    def _generate_cluster_theme(self, concepts: Dict[str, List[str]], chunks: List[str]) -> str:
        """Generate a descriptive theme name for a cluster."""
        # Priority order for theme naming
        theme_priorities = ['languages', 'behaviors', 'actions', 'constraints', 'topics']
        
        for priority in theme_priorities:
            if priority in concepts and concepts[priority]:
                items = concepts[priority]
                if len(items) == 1:
                    return f"{priority.title()}: {items[0].title()}"
                elif len(items) <= 3:
                    return f"{priority.title()}: {', '.join([item.title() for item in items[:2]])}"
                else:
                    return f"{priority.title()} ({len(items)} items)"
        
        # Fallback: use common words from chunks
        all_words = []
        for chunk in chunks:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', chunk.lower())
            all_words.extend(words)
        
        # Find most common meaningful word
        word_counts = {}
        stop_words = {'the', 'and', 'you', 'are', 'should', 'must', 'can', 'will', 'have', 'been', 'this', 'that'}
        for word in all_words:
            if word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        if word_counts:
            most_common = max(word_counts.items(), key=lambda x: x[1])[0]
            return f"Instructions: {most_common.title()}"
        
        return f"Mixed Instructions"
    
    def generate_cluster_insights(self, chunks: List[str], clustering_results: Dict) -> List[str]:
        """Generate actionable insights about prompt clustering and its impact on LLM output."""
        insights = []
        clusters = clustering_results['clusters']
        analysis = clustering_results['cluster_analysis']
        n_clusters = clustering_results['n_clusters']
        silhouette = clustering_results['silhouette_score']
        
        # Overall structure insight
        if n_clusters == 1:
            insights.append("🎯 **Focused Prompt**: All instructions belong to one theme. This creates consistent, predictable LLM behavior.")
        elif n_clusters <= 3:
            insights.append(f"✅ **Well-Organized**: {n_clusters} distinct instruction themes detected. This balance helps the LLM understand different aspects without confusion.")
        else:
            insights.append(f"⚠️ **Complex Structure**: {n_clusters} different themes detected. Consider grouping related instructions to reduce cognitive load on the LLM.")
        
        # Clustering quality insight
        if silhouette > 0.5:
            insights.append(f"🔍 **Clear Separation**: High cluster quality (score: {silhouette:.2f}). Instructions are well-differentiated, reducing ambiguity for the LLM.")
        elif silhouette > 0.2:
            insights.append(f"📊 **Moderate Separation**: Decent cluster quality (score: {silhouette:.2f}). Some instruction overlap may cause minor confusion.")
        else:
            insights.append(f"🌀 **Overlapping Instructions**: Low cluster quality (score: {silhouette:.2f}). Mixed themes may lead to inconsistent LLM responses.")
        
        # Individual cluster insights
        for cluster_id, cluster_info in analysis.items():
            theme = cluster_info['theme']
            size = cluster_info['size']
            cohesion = cluster_info['cohesion']
            
            if size == 1:
                insights.append(f"🔸 **Isolated Theme - {theme}**: Single instruction may need reinforcement or integration with related themes.")
            elif cohesion > 0.7:
                insights.append(f"🔹 **Strong Theme - {theme}** ({size} chunks): High coherence ({cohesion:.2f}) will create consistent LLM behavior in this area.")
            elif cohesion < 0.4:
                insights.append(f"🔸 **Weak Theme - {theme}** ({size} chunks): Low coherence ({cohesion:.2f}) may cause conflicting LLM responses.")
        
        # Specific prompt optimization suggestions
        language_clusters = [c for c in analysis.values() if 'language' in c['theme'].lower()]
        behavior_clusters = [c for c in analysis.values() if any(word in c['theme'].lower() for word in ['behavior', 'nice', 'helpful', 'professional'])]
        
        if len(language_clusters) > 1:
            insights.append("💡 **Language Optimization**: Multiple language instruction clusters detected. Consider consolidating to avoid LLM confusion about which language to prioritize.")
        
        if len(behavior_clusters) > 1:
            insights.append("💡 **Behavior Optimization**: Multiple behavior instruction clusters found. Merging these could create more consistent personality in LLM responses.")
        
        # Impact on LLM attention
        largest_cluster = max(analysis.values(), key=lambda x: x['size'])
        if largest_cluster['size'] > len(chunks) * 0.6:
            insights.append(f"⭐ **Dominant Theme**: '{largest_cluster['theme']}' dominates your prompt ({largest_cluster['size']}/{len(chunks)} chunks). This theme will strongly influence LLM behavior.")
        
        return insights


@dataclass
class CognitiveLoadResult:
    """Results of cognitive load analysis."""
    overall_load: float  # 0-100 scale
    risk_level: str     # Low, Medium, High, Critical
    breakdown: Dict[str, float]
    conflicts: List[str]
    recommendations: List[str]
    load_heatmap: np.ndarray


class CognitiveLoadAnalyzer:
    """Analyzes cognitive load of system prompts based on semantic complexity."""
    
    def __init__(self):
        # Define cognitive domain categories
        self.cognitive_domains = {
            'language': ['spanish', 'english', 'hebrew', 'french', 'translate', 'language'],
            'behavior': ['helpful', 'nice', 'professional', 'friendly', 'courteous', 'polite'],
            'reasoning': ['analyze', 'think', 'logic', 'reason', 'conclude', 'infer'],
            'creativity': ['creative', 'innovative', 'original', 'imaginative', 'artistic'],
            'memory': ['remember', 'recall', 'store', 'memorize', 'forget', 'retain'],
            'formatting': ['format', 'structure', 'organize', 'layout', 'style'],
            'constraints': ['never', 'always', 'only', 'must', 'forbidden', 'required'],
            'math': ['calculate', 'compute', 'math', 'number', 'equation', 'formula'],
            'communication': ['respond', 'answer', 'reply', 'communicate', 'speak', 'write']
        }
        
        # Define contradiction patterns
        self.contradiction_patterns = [
            (r'\balways\b', r'\bnever\b'),
            (r'\bonly\b', r'\bnever\b'),
            (r'\bmust\b', r'\bforbidden\b|prohibited\b'),
            (r'\benglish\b', r'\bnever.*english\b|only.*spanish\b'),
            (r'\bspanish\b', r'\bnever.*spanish\b|only.*english\b'),
            (r'\bformal\b', r'\binformal\b|casual\b'),
            (r'\bshort\b', r'\blong\b|detailed\b'),
            (r'\bquick\b', r'\bslow\b|careful\b')
        ]
    
    def analyze_cognitive_load(self, chunks: List[str], clustering_results: Dict, 
                             similarity_matrix: np.ndarray, attention_scores: np.ndarray = None) -> CognitiveLoadResult:
        """Perform comprehensive cognitive load analysis."""
        
        # Calculate individual load components
        conflict_load = self._calculate_semantic_conflict_load(chunks, clustering_results)
        domain_load = self._calculate_domain_diversity_load(chunks, clustering_results)
        switching_load = self._calculate_context_switching_load(clustering_results, similarity_matrix)
        memory_load = self._calculate_working_memory_load(chunks, clustering_results)
        attention_load = self._calculate_attention_fragmentation_load(attention_scores) if attention_scores is not None else 0
        
        # Weighted overall cognitive load (0-100 scale)
        overall_load = (
            conflict_load * 0.35 +      # 35% - conflicts are most critical
            domain_load * 0.20 +        # 20% - domain diversity
            switching_load * 0.20 +     # 20% - context switching cost
            memory_load * 0.15 +        # 15% - working memory requirements
            attention_load * 0.10       # 10% - attention distribution
        )
        
        # Assess risk level
        risk_level = self._assess_risk_level(overall_load)
        
        # Generate specific conflict descriptions
        conflicts = self._identify_specific_conflicts(chunks, clustering_results)
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations(
            overall_load, conflict_load, domain_load, switching_load, memory_load
        )
        
        # Create load heatmap between clusters
        load_heatmap = self._create_cognitive_load_heatmap(clustering_results, similarity_matrix)
        
        return CognitiveLoadResult(
            overall_load=overall_load,
            risk_level=risk_level,
            breakdown={
                'semantic_conflicts': conflict_load,
                'domain_diversity': domain_load, 
                'context_switching': switching_load,
                'working_memory': memory_load,
                'attention_fragmentation': attention_load
            },
            conflicts=conflicts,
            recommendations=recommendations,
            load_heatmap=load_heatmap
        )
    
    def _calculate_semantic_conflict_load(self, chunks: List[str], clustering_results: Dict) -> float:
        """Calculate load from semantic contradictions between instructions."""
        conflict_score = 0.0
        total_comparisons = 0
        
        clusters = clustering_results['clusters']
        
        # Check for conflicts between all chunk pairs
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                chunk_i = chunks[i].lower()
                chunk_j = chunks[j].lower()
                
                # Check each contradiction pattern
                for pattern1, pattern2 in self.contradiction_patterns:
                    match1_i = bool(re.search(pattern1, chunk_i))
                    match2_i = bool(re.search(pattern2, chunk_i))
                    match1_j = bool(re.search(pattern1, chunk_j))
                    match2_j = bool(re.search(pattern2, chunk_j))
                    
                    # Direct contradiction: chunk i has pattern1, chunk j has pattern2
                    if (match1_i and match2_j) or (match2_i and match1_j):
                        # Weight conflicts more heavily if they're in different clusters
                        cluster_i = self._find_chunk_cluster(i, clusters)
                        cluster_j = self._find_chunk_cluster(j, clusters)
                        
                        if cluster_i != cluster_j:
                            conflict_score += 2.0  # Inter-cluster conflicts are worse
                        else:
                            conflict_score += 1.0  # Intra-cluster conflicts
                
                total_comparisons += 1
        
        # Normalize to 0-100 scale
        if total_comparisons > 0:
            normalized_score = min(100, (conflict_score / total_comparisons) * 50)
        else:
            normalized_score = 0
            
        return normalized_score
    
    def _calculate_domain_diversity_load(self, chunks: List[str], clustering_results: Dict) -> float:
        """Calculate cognitive load from handling multiple conceptual domains."""
        active_domains = set()
        
        # Find which cognitive domains are active in the prompt
        for chunk in chunks:
            chunk_lower = chunk.lower()
            for domain, keywords in self.cognitive_domains.items():
                for keyword in keywords:
                    if keyword in chunk_lower:
                        active_domains.add(domain)
                        break
        
        domain_count = len(active_domains)
        
        # Cognitive load increases non-linearly with domain count
        if domain_count <= 2:
            load = domain_count * 10  # Low load: 10-20
        elif domain_count <= 4:
            load = 20 + (domain_count - 2) * 20  # Medium load: 40-60
        else:
            load = 60 + (domain_count - 4) * 15  # High load: 75+
            
        return min(100, load)
    
    def _calculate_context_switching_load(self, clustering_results: Dict, similarity_matrix: np.ndarray) -> float:
        """Calculate cognitive cost of switching between different cluster contexts."""
        clusters = clustering_results['clusters']
        n_clusters = len(clusters)
        
        if n_clusters <= 1:
            return 0.0
        
        # Calculate average dissimilarity between clusters
        cluster_distances = []
        
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                cluster_i_indices = clusters[i]
                cluster_j_indices = clusters[j]
                
                # Calculate average similarity between chunks in different clusters
                similarities = []
                for idx_i in cluster_i_indices:
                    for idx_j in cluster_j_indices:
                        if idx_i < len(similarity_matrix) and idx_j < len(similarity_matrix[0]):
                            similarities.append(similarity_matrix[idx_i][idx_j])
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    distance = 1 - avg_similarity  # Convert similarity to distance
                    cluster_distances.append(distance)
        
        if cluster_distances:
            avg_distance = np.mean(cluster_distances)
            # Higher average distance = higher switching cost
            switching_load = avg_distance * 100
        else:
            switching_load = 0
            
        return min(100, switching_load)
    
    def _calculate_working_memory_load(self, chunks: List[str], clustering_results: Dict) -> float:
        """Calculate cognitive load from number of distinct concepts to track."""
        unique_concepts = set()
        
        # Extract unique semantic concepts from all chunks
        for chunk in chunks:
            chunk_lower = chunk.lower()
            # Extract key terms (nouns, adjectives, verbs)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', chunk_lower)
            
            # Filter for meaningful content words
            meaningful_words = [
                word for word in words 
                if word not in ['the', 'and', 'you', 'are', 'should', 'must', 'can', 'will', 'have', 'from', 'time']
            ]
            
            unique_concepts.update(meaningful_words)
        
        concept_count = len(unique_concepts)
        
        # Working memory load based on cognitive psychology research
        # Human working memory: ~7±2 items, LLM probably similar constraints
        if concept_count <= 5:
            load = concept_count * 8  # Low load: 8-40
        elif concept_count <= 10:
            load = 40 + (concept_count - 5) * 10  # Medium load: 50-90
        else:
            load = 90 + (concept_count - 10) * 2  # High load: 92+
            
        return min(100, load)
    
    def _calculate_attention_fragmentation_load(self, attention_scores: np.ndarray) -> float:
        """Calculate load from fragmented attention distribution."""
        if attention_scores is None or len(attention_scores) == 0:
            return 0.0
        
        # Calculate entropy of attention distribution
        # High entropy = fragmented attention = higher cognitive load
        normalized_scores = attention_scores / np.sum(attention_scores)
        entropy = -np.sum(normalized_scores * np.log2(normalized_scores + 1e-10))
        
        # Normalize entropy to 0-100 scale
        max_entropy = np.log2(len(attention_scores))  # Maximum possible entropy
        fragmentation_load = (entropy / max_entropy) * 100 if max_entropy > 0 else 0
        
        return fragmentation_load
    
    def _find_chunk_cluster(self, chunk_index: int, clusters: Dict) -> int:
        """Find which cluster a chunk belongs to."""
        for cluster_id, chunk_indices in clusters.items():
            if chunk_index in chunk_indices:
                return cluster_id
        return -1  # Not found
    
    def _assess_risk_level(self, overall_load: float) -> str:
        """Assess degradation risk level based on overall cognitive load."""
        if overall_load >= 80:
            return "Critical"
        elif overall_load >= 60:
            return "High"
        elif overall_load >= 40:
            return "Medium"
        else:
            return "Low"
    
    def _identify_specific_conflicts(self, chunks: List[str], clustering_results: Dict) -> List[str]:
        """Identify and describe specific conflicts found."""
        conflicts = []
        
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                chunk_i = chunks[i]
                chunk_j = chunks[j]
                
                # Check for direct contradictions
                for pattern1, pattern2 in self.contradiction_patterns:
                    if (re.search(pattern1, chunk_i.lower()) and re.search(pattern2, chunk_j.lower())) or \
                       (re.search(pattern2, chunk_i.lower()) and re.search(pattern1, chunk_j.lower())):
                        conflicts.append(f"Conflict between Chunk {i+1} and Chunk {j+1}: "
                                       f"'{chunk_i[:50]}...' contradicts '{chunk_j[:50]}...'")
        
        return conflicts[:5]  # Limit to top 5 conflicts
    
    def _generate_optimization_recommendations(self, overall_load: float, conflict_load: float, 
                                             domain_load: float, switching_load: float, 
                                             memory_load: float) -> List[str]:
        """Generate specific optimization recommendations."""
        recommendations = []
        
        if conflict_load > 50:
            recommendations.append("🔴 CRITICAL: Remove contradictory instructions that confuse the LLM")
        
        if domain_load > 60:
            recommendations.append("⚠️ Consider splitting prompt into domain-specific sections")
        
        if switching_load > 70:
            recommendations.append("💡 Group similar concepts together to reduce context switching")
            
        if memory_load > 80:
            recommendations.append("🧠 Simplify or consolidate instructions to reduce working memory load")
        
        if overall_load > 70:
            recommendations.append("🚨 HIGH RISK: This prompt may cause inconsistent LLM behavior")
        elif overall_load < 30:
            recommendations.append("✅ Low cognitive load - prompt should perform consistently")
        
        return recommendations
    
    def _create_cognitive_load_heatmap(self, clustering_results: Dict, similarity_matrix: np.ndarray) -> np.ndarray:
        """Create a heatmap showing cognitive interference between clusters."""
        clusters = clustering_results['clusters']
        n_clusters = len(clusters)
        
        load_matrix = np.zeros((n_clusters, n_clusters))
        
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i == j:
                    load_matrix[i][j] = 0  # No self-interference
                else:
                    # Calculate interference based on dissimilarity
                    cluster_i_indices = clusters[i]
                    cluster_j_indices = clusters[j]
                    
                    similarities = []
                    for idx_i in cluster_i_indices:
                        for idx_j in cluster_j_indices:
                            if idx_i < len(similarity_matrix) and idx_j < len(similarity_matrix[0]):
                                similarities.append(similarity_matrix[idx_i][idx_j])
                    
                    if similarities:
                        avg_similarity = np.mean(similarities)
                        # High dissimilarity = high cognitive interference
                        interference = (1 - avg_similarity) * 100
                        load_matrix[i][j] = interference
        
        return load_matrix
