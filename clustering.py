"""Utility for clustering high-dimensional embeddings and visualizing the results.

This module provides a helper to cluster embedding vectors using different
scikit-learn algorithms and display the clusters using t-SNE for dimensionality
reduction.
"""

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.manifold import TSNE

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
    title: Optional[str] = None,
) -> plt.Figure:
    """Create a t-SNE scatter plot of clustered embeddings.

    The returned matplotlib figure can be rendered in different contexts,
    such as Streamlit via ``st.pyplot``. Perplexity is adjusted to remain
    valid for the number of provided samples.
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        raise ValueError("At least two samples are required for t-SNE visualization.")

    perplexity = min(30, n_samples - 1)
    reduced = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(
        embeddings
    )
    fig, ax = plt.subplots()
    ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10")
    ax.set_xlabel("TSNE-1")
    ax.set_ylabel("TSNE-2")
    ax.set_title(title or "Embedding clusters")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage with the Iris dataset.
    from sklearn.datasets import load_iris

    iris = load_iris()
    data = iris.data
    labels = cluster_embeddings(data, algorithm="kmeans", n_clusters=3)
    fig = visualize_clusters(data, labels, title="KMeans clustering of Iris data")
    fig.show()
