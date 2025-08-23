import numpy as np
from matplotlib.figure import Figure

from clustering import build_chunk_graph, visualize_clusters


def test_visualize_clusters_uses_pca():
    """visualize_clusters should always use PCA for dimensionality reduction."""
    embeddings = np.array([[float(i), float(i)] for i in range(5)])
    labels = [0, 1, 0, 1, 0]
    fig = visualize_clusters(embeddings, labels)
    assert isinstance(fig, Figure)
    assert "PCA" in fig.axes[0].get_title()


def test_build_chunk_graph_creates_edges_for_similar_chunks():
    chunks = ["alpha", "alpha variant", "beta"]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    graph = build_chunk_graph(chunks, embeddings, threshold=0.85)
    assert graph.number_of_nodes() == 3
    # First two chunks are similar; third is different
    assert graph.has_edge(0, 1)
    assert not graph.has_edge(0, 2)
