"""Utility for clustering high-dimensional embeddings and visualizing the results.

This module provides a helper to cluster embedding vectors using different
scikit-learn algorithms and display the clusters using PCA for dimensionality
reduction.
"""

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Mapping of algorithm names to their constructors. Functions accept the desired
# number of clusters and return an instantiated estimator.
CLUSTERERS = {
    "kmeans": lambda n: KMeans(n_clusters=n, random_state=42),
    "agglomerative": lambda n: AgglomerativeClustering(n_clusters=n),
    "dbscan": lambda _: DBSCAN(),
}


def cluster_embeddings(
    embeddings: np.ndarray,
    algorithm: str = "kmeans",
    n_clusters: int = 3,
) -> np.ndarray:
    """Cluster embedding vectors using the selected algorithm.

    Parameters
    ----------
    embeddings:
        Array of shape (n_samples, n_features) containing the embeddings.
    algorithm:
        Name of the clustering algorithm to use. Options are ``"kmeans"``,
        ``"agglomerative"``, and ``"dbscan"``.
    n_clusters:
        Number of clusters for algorithms that require it.

    Returns
    -------
    np.ndarray
        The cluster label for each embedding.
    """
    algorithm = algorithm.lower()
    if algorithm not in CLUSTERERS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    clusterer = CLUSTERERS[algorithm](n_clusters)
    labels = clusterer.fit_predict(embeddings)
    return labels


def visualize_clusters(
    embeddings: np.ndarray,
    labels: Iterable[int],
    title: str = "Sentence clusters",
    random_state: int = 42,
) -> plt.Figure:
    """Visualize clustered embeddings using PCA.

    Parameters
    ----------
    embeddings:
        Array of shape ``(n_samples, n_features)`` with the embedding vectors.
    labels:
        Iterable of cluster labels for each embedding.
    title:
        Plot title. Defaults to ``"Sentence clusters"``.
    random_state:
        Random seed for reproducible dimensionality reduction.

    Returns
    -------
    matplotlib.figure.Figure
        The generated scatter plot.
    """
    X = np.asarray(embeddings)
    n_samples = int(X.shape[0])

    if n_samples < 2:
        raise ValueError("At least two samples are required for visualization.")

    if np.isnan(X).any() or np.isinf(X).any():
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X = X[valid_mask]
        labels = [labels[i] for i, ok in enumerate(valid_mask) if ok]
        n_samples = int(X.shape[0])
        if n_samples < 2:
            raise ValueError("Not enough valid samples after removing NaN/Inf rows.")

    reducer = PCA(n_components=2, random_state=random_state)
    Y = reducer.fit_transform(X)
    method = "PCA"

    fig, ax = plt.subplots()
    ax.scatter(Y[:, 0], Y[:, 1], s=40)
    ax.set_title(f"{title} • {method}")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.3)
    return fig


def build_chunk_graph(
    chunks: Iterable[str],
    embeddings: np.ndarray,
    threshold: float = 0.8,
) -> nx.Graph:
    """Create a graph connecting similar chunks based on cosine similarity.

    Parameters
    ----------
    chunks:
        Iterable of text chunks.
    embeddings:
        Array of shape ``(n_samples, n_features)`` containing chunk embeddings.
    threshold:
        Minimum cosine similarity required to create an edge between two chunks.

    Returns
    -------
    networkx.Graph
        Undirected graph with one node per chunk and edges for similar chunks.
    """
    texts = list(chunks)
    X = np.asarray(embeddings)
    if X.shape[0] != len(texts):
        raise ValueError("Number of chunks and embeddings must match")

    graph = nx.Graph()
    for idx, text in enumerate(texts):
        graph.add_node(idx, text=text)

    sim = cosine_similarity(X)
    n = sim.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                graph.add_edge(i, j, weight=float(sim[i, j]))
    return graph


if __name__ == "__main__":
    # Example usage with the Iris dataset.
    from sklearn.datasets import load_iris

    iris = load_iris()
    data = iris.data
    labels = cluster_embeddings(data, algorithm="kmeans", n_clusters=3)
    fig = visualize_clusters(data, labels, title="KMeans clustering of Iris data")
    fig.show()
