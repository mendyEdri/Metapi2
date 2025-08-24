import numpy as np
import pytest
from matplotlib.figure import Figure

from clustering import (
    build_chunk_graph,
    compute_chunk_weights,
    rank_chunks,
    visualize_clusters,
)


def test_visualize_clusters_uses_pca():
    """visualize_clusters should always use PCA for dimensionality reduction."""
    embeddings = np.array([[float(i), float(i)] for i in range(5)])
    labels = [0, 1, 0, 1, 0]
    fig = visualize_clusters(embeddings, labels)
    assert isinstance(fig, Figure)
    assert "PCA" in fig.axes[0].get_title()


def test_visualize_clusters_labels_points_with_chunk_indices():
    embeddings = np.array([[float(i), float(i)] for i in range(3)])
    # deliberately give repeated cluster labels to ensure chunk indices are used
    labels = [0, 0, 1]
    fig = visualize_clusters(embeddings, labels)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert len(texts) == len(labels)
    # Expect the annotations to be the chunk indices 0,1,2 regardless of labels
    assert set(texts) == {"0", "1", "2"}


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


def test_compute_chunk_weights_center_downweights_outlier():
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    weights = compute_chunk_weights(embeddings, method="center")
    assert np.isclose(weights.sum(), 1.0)
    # the outlier opposite to the center should receive the lowest weight
    assert weights[3] < weights[0]
    assert np.allclose(weights[:3], weights[0], atol=1e-6)


def test_rank_chunks_pagerank_highlights_connected_nodes():
    chunks = ["alpha", "alpha variant", "beta"]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    graph = build_chunk_graph(chunks, embeddings, threshold=0.85)
    ranking = rank_chunks(graph, method="pagerank")
    # first two nodes are connected and should outrank the isolated third node
    assert ranking[0][0] in {0, 1}
    assert ranking[-1][0] == 2


def test_rank_chunks_rejects_unknown_method():
    chunks = ["alpha", "beta"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    graph = build_chunk_graph(chunks, embeddings, threshold=0.5)
    with pytest.raises(ValueError):
        rank_chunks(graph, method="unknown")
