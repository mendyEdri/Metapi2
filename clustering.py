"""Utility for clustering high-dimensional embeddings and visualizing the results.

This module provides a helper to cluster embedding vectors using different
scikit-learn algorithms and display the clusters using PCA for dimensionality
reduction.
"""

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

# ``networkx`` is an optional dependency. Importing it lazily ensures that
# consumers who only need clustering utilities (e.g. ``compute_chunk_weights``)
# can still use this module even if ``networkx`` isn't installed. The graph
# building helper will raise a clear error when ``networkx`` is unavailable.
try:  # pragma: no cover - tested indirectly via build_chunk_graph
    import networkx as nx
except ModuleNotFoundError:  # pragma: no cover - exercised when networkx missing
    nx = None
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

    Notes
    -----
    Points are colored according to their cluster label and annotated with
    their chunk index to ease interpretation.
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
    labels_arr = np.asarray(list(labels))
    ax.scatter(Y[:, 0], Y[:, 1], c=labels_arr, cmap="tab10", s=40)
    for idx, (x, y) in enumerate(Y):
        # annotate each point with its chunk index above the marker
        ax.text(x, y, str(idx), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"{title} • {method}")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.3)
    return fig


def compute_chunk_weights(
    embeddings: np.ndarray,
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute softmax-normalized weights for each embedding.

    The weight for a chunk is derived from its cosine similarity to a reference
    vector. When ``reference`` is ``None`` the centroid of all embeddings is
    used. The similarities are converted to a probability distribution via a
    softmax so that the weights sum to 1.

    Parameters
    ----------
    embeddings:
        Array of shape ``(n_samples, n_features)`` containing the embeddings.
    reference:
        Optional reference vector. If ``None`` the centroid of ``embeddings``
        is used.

    Returns
    -------
    np.ndarray
        Array of weights with the same length as ``embeddings``.
    """
    X = np.asarray(embeddings)
    if X.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    ref = np.asarray(reference) if reference is not None else X.mean(axis=0)
    ref = ref.reshape(1, -1)
    sims = cosine_similarity(X, ref).ravel()

    # Numerical stability: subtract max before exponentiating
    exp_scores = np.exp(sims - np.max(sims))
    weights = exp_scores / exp_scores.sum()
    return weights


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
    if nx is None:  # pragma: no cover - exercised when networkx missing
        raise ImportError(
            "build_chunk_graph requires the 'networkx' package. Install it via"
            " 'pip install networkx' to enable graph construction."
        )

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

# Public API for ``from clustering import *``. Keeping the ``__all__``
# declaration at the end of the module ensures that any helper defined above
# (such as ``compute_chunk_weights``) is always exported.
__all__ = [
    "cluster_embeddings",
    "visualize_clusters",
    "compute_chunk_weights",
    "build_chunk_graph",
]
