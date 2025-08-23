import numpy as np
from matplotlib.figure import Figure

from clustering import build_chunk_graph, visualize_clusters, compute_chunk_weights


def test_visualize_clusters_uses_pca():
    """visualize_clusters should always use PCA for dimensionality reduction."""
    embeddings = np.array([[float(i), float(i)] for i in range(5)])
    labels = [0, 1, 0, 1, 0]
    fig = visualize_clusters(embeddings, labels)
    assert isinstance(fig, Figure)
    assert "PCA" in fig.axes[0].get_title()


def test_visualize_clusters_labels_points_with_cluster_numbers():
    embeddings = np.array([[float(i), float(i)] for i in range(3)])
    labels = [0, 1, 2]
    fig = visualize_clusters(embeddings, labels)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert len(texts) == len(labels)
    assert set(texts) == {str(l) for l in labels}


def test_build_chunk_graph_creates_edges_for_similar_chunks():
    chunks = ["alpha", "alpha variant", "beta"]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    graph = build_chunk_graph(chunks, embeddings, threshold=0.85)
    assert graph.number_of_nodes() == 3
    # First two chunks are similar; third is different
    assert graph.has_edge(0, 1)
    assert not graph.has_edge(0, 2)


def test_compute_chunk_weights_emphasizes_similar_chunks():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    reference = np.array([1.0, 0.0])
    weights = compute_chunk_weights(embeddings, reference=reference)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] == weights[2] > weights[1]
